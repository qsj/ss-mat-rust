use std::collections::HashMap;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use jvm_hprof::heap_dump::PrimitiveArrayType;
use jvm_hprof::{heap_dump, IdSize, RecordTag};
use memmap2::Mmap;
use rayon::prelude::*;

use ss_heap_model::class_info::{ClassCategory, JavaClassInfo};
use ss_heap_model::heap_dump::{FieldValue, FrameLocals, HeapDump, LocalVarDetail};
use ss_stats::engine::StatsEngine;

/// Tunables for problem detection during the linear scan.
#[derive(Clone, Copy, Debug)]
pub struct ScanConfig {
    pub min_trailing_zeroes_percent: f64,
    pub min_sparse_zeroes_percent: f64,
    pub min_unused_hi_bits_percent: f64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            min_trailing_zeroes_percent: 25.0,
            min_sparse_zeroes_percent: 70.0,
            min_unused_hi_bits_percent: 70.0,
        }
    }
}

/// Configure rayon's global pool when a thread limit is requested.
/// Returns an error if the pool was already initialized by another caller.

/// Hand the mapped dump's pages back to the kernel.
///
/// The scan keeps the whole dump mapped so object bodies can be borrowed instead
/// of copied, which is what makes the phase cheap; but those file pages otherwise
/// stay resident and dominate RSS. Releasing them is safe: any later access simply
/// faults the page back in from the read-only mapping.
fn release_mapped_pages(map: &memmap2::Mmap) {
    if map.is_empty() {
        return;
    }
    // SAFETY: the range is the whole read-only file mapping; re-reading a released
    // page only re-faults it from the file, and no writer is involved.
    let _ = unsafe {
        map.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, 0, map.len())
    };
}

// ---------------------------------------------------------------------------
// Random access to object bodies
// ---------------------------------------------------------------------------
//
// The extraction walks below ask for a handful of objects by address: a thread
// and the Strings it names, the Properties object graph, an OOM thread's locals,
// and the objects one reference chain renders. Those bodies are read with pread
// into a small buffer instead of mapping the dump, so the dump's pages never land
// in this process's resident set and no per-object payload outlives the call that
// used it.

/// Object kinds stored in the object directory.
const DI_KIND_INSTANCE: u8 = 0;
const DI_KIND_OBJ_ARRAY: u8 = 1;
const DI_KIND_PRIM_ARRAY: u8 = 2;
const DI_KIND_CLASS: u8 = 3;

/// Bytes read at once for one class dump; every dump is far smaller.
const CLASS_DUMP_WINDOW: usize = 64 << 10;

fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        match file.read_at(&mut buf[done..], offset + done as u64) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read from the heap dump",
                ))
            }
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn read_some_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    let mut done = 0usize;
    while done < buf.len() {
        match file.read_at(&mut buf[done..], offset + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// One object's payload, read out of the dump.
struct Body {
    /// Class index the parser assigned (u32::MAX when unresolved).
    class_idx: u32,
    /// One of DI_KIND_INSTANCE / DI_KIND_OBJ_ARRAY / DI_KIND_PRIM_ARRAY.
    kind: u8,
    /// Element width in bytes for arrays.
    elem: u8,
    /// Element count for arrays.
    len: u32,
    /// HPROF basic type code for primitive arrays.
    prim_type: u8,
    /// Instance field bytes, or the array's raw payload.
    bytes: Vec<u8>,
}

/// Address to body lookups backed by the on-disk object directory.
struct AddrBodies<'a> {
    heap: &'a HeapDump,
    dir: &'a ss_support::obj_dir::ObjDir,
    file: std::fs::File,
    id_bytes: usize,
}

impl<'a> AddrBodies<'a> {
    /// Open the dump for random body reads; None when the run has no object
    /// directory or the dump cannot be reopened.
    fn open(heap: &'a HeapDump, hprof_path: &Path) -> Option<Self> {
        let dir = heap.obj_dir.as_ref()?;
        let file = std::fs::File::open(hprof_path).ok()?;
        Some(Self { heap, dir, file, id_bytes: heap.identifier_size as usize })
    }

    /// Read one object's payload. Class dumps have no payload and report None.
    fn get(&self, addr: u64) -> Option<Body> {
        let rec = self.dir.lookup(addr).ok()??.1;
        let bytes = self.body_bytes(&rec)?;
        Some(Body {
            class_idx: rec.class_idx,
            kind: rec.kind,
            elem: rec.elem,
            len: rec.len,
            prim_type: rec.flags as u8,
            bytes,
        })
    }

    fn body_bytes(&self, rec: &ss_support::obj_dir::ObjRec) -> Option<Vec<u8>> {
        let len = match rec.kind {
            DI_KIND_INSTANCE => {
                // The declared field-byte count sits immediately before the body.
                if rec.file_offset < 4 {
                    return None;
                }
                let mut head = [0u8; 4];
                read_exact_at(&self.file, &mut head, rec.file_offset - 4).ok()?;
                u32::from_be_bytes(head) as u64
            }
            DI_KIND_OBJ_ARRAY => rec.len as u64 * self.id_bytes as u64,
            DI_KIND_PRIM_ARRAY => rec.len as u64 * rec.elem as u64,
            _ => return None,
        };
        let len = usize::try_from(len).ok()?;
        if len == 0 {
            return Some(Vec::new());
        }
        let mut buf = vec![0u8; len];
        read_exact_at(&self.file, &mut buf, rec.file_offset).ok()?;
        Some(buf)
    }

    /// Object id held by the static field field_name of the class dump at
    /// class_obj_id. The same pass that builds the object directory records the
    /// class-dump body offset, so this is one small read instead of a scan.
    fn class_static_ref(
        &self,
        class_obj_id: u64,
        field_name: &str,
        string_table: &HashMap<u64, String>,
    ) -> Option<u64> {
        let rec = self.dir.lookup(class_obj_id).ok()??.1;
        if rec.kind != DI_KIND_CLASS || rec.file_offset == 0 {
            return None;
        }
        let mut buf = vec![0u8; CLASS_DUMP_WINDOW];
        let got = read_some_at(&self.file, &mut buf, rec.file_offset).ok()?;
        buf.truncate(got);
        let id_size = self.id_bytes as u8;
        let mut cur = WindowCursor::new(&buf, id_size);
        cur.id()?; // class object id
        cur.u32()?; // stack trace serial
        cur.id()?; // super class
        cur.skip(5 * id_size as u64)?; // loader, signers, protection domain, 2 reserved
        cur.u32()?; // instance size
        for _ in 0..cur.u16()? {
            cur.skip(2)?;
            let ty = cur.u8()?;
            cur.skip(field_type_size(ty, id_size as usize) as u64)?;
        }
        // The last matching static field wins, matching the streaming collector.
        let mut found: Option<Option<u64>> = None;
        for _ in 0..cur.u16()? {
            let name_id = cur.id()?;
            let ty = cur.u8()?;
            let width = field_type_size(ty, id_size as usize);
            let value = cur.take(width)?;
            if ty == 2 && string_table.get(&name_id).map(|s| s.as_str()) == Some(field_name) {
                let id = if width == 4 {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(value);
                    u32::from_be_bytes(b) as u64
                } else {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(value);
                    u64::from_be_bytes(b)
                };
                found = Some(if id == 0 { None } else { Some(id) });
            }
        }
        found.flatten()
    }
}

/// Bounds-checked big-endian cursor over a read window.
struct WindowCursor<'a> {
    data: &'a [u8],
    pos: usize,
    id_size: u8,
}

impl<'a> WindowCursor<'a> {
    fn new(data: &'a [u8], id_size: u8) -> Self {
        Self { data, pos: 0, id_size }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let out = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(out)
    }

    fn skip(&mut self, n: u64) -> Option<()> {
        self.take(usize::try_from(n).ok()?).map(|_| ())
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        let b = self.take(2)?;
        Some(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        let b = self.take(4)?;
        Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn id(&mut self) -> Option<u64> {
        if self.id_size == 4 {
            return Some(self.u32()? as u64);
        }
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Some(u64::from_be_bytes(a))
    }
}

pub fn configure_thread_pool(max_threads: i32) -> Result<()> {
    if max_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads as usize)
            .build_global()
            .map_err(|e| anyhow::anyhow!("Failed to configure scan threads: {}", e))?;
    }
    Ok(())
}

/// Linear scan coordinator: re-parses the HPROF to access field data
/// and runs 4 parallel scan passes over instances and arrays.
pub struct LinearScanCoordinator {
    heap: Arc<HeapDump>,
    stats: Arc<StatsEngine>,
    config: ScanConfig,
}

impl LinearScanCoordinator {
    pub fn new(heap: Arc<HeapDump>, stats: Arc<StatsEngine>) -> Self {
        Self { heap, stats, config: ScanConfig::default() }
    }

    pub fn with_config(heap: Arc<HeapDump>, stats: Arc<StatsEngine>, config: ScanConfig) -> Self {
        Self { heap, stats, config }
    }

    pub fn stats(&self) -> Arc<StatsEngine> {
        Arc::clone(&self.stats)
    }

    /// Run all scan passes. Re-parses the HPROF to access field-level data.
    ///
    /// Every pass is sequential over the mapped dump and keeps only aggregates,
    /// the report's top-N detail lists and fixed-size buffers: per-object payloads
    /// (string contents, array bodies, element vectors) are never materialised,
    /// they are re-read from the mapping by offset.
    pub fn run(&self, hprof_path: &Path) -> Result<()> {
        eprintln!("Linear scan: re-parsing HPROF for field data...");
        ss_support::mem_trace::trace("run-start");

        let file = std::fs::File::open(hprof_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data = &mmap[..];

        let hprof = jvm_hprof::parse_hprof(data)
            .map_err(|e| anyhow::anyhow!("Failed to re-parse HPROF: {:?}", e))?;
        let id_size = hprof.header().id_size();
        // Publish the id size so the DirectBuffer field readers can decode
        // address/long fields with the correct width.
        ID_SIZE.with(|c| c.set(match id_size { IdSize::U32 => 4, IdSize::U64 => 8 }));
        let id_bytes = match id_size { IdSize::U32 => 4usize, IdSize::U64 => 8usize };
        let walk_id_size = id_bytes as u8;
        drop(hprof);

        // Class object address -> class index.
        let mut class_obj_ids: HashMap<u64, u32> = HashMap::with_capacity(self.heap.classes.len());
        for cls in &self.heap.classes {
            if cls.class_object_id != 0 {
                class_obj_ids.insert(cls.class_object_id, cls.index);
            }
        }

        // Java's alignedBodySize = align(header + total field sizes over the
        // class hierarchy), not HPROF's padded instance_size.
        let header_sz = self.heap.header_size as i64;
        let mut class_aligned = vec![0i64; self.heap.classes.len()];
        for cls in &self.heap.classes {
            let mut field_bytes = 0i64;
            let mut current_id = cls.class_object_id;
            let mut seen = std::collections::HashSet::new();
            loop {
                let cidx = match class_obj_ids.get(&current_id).copied() {
                    Some(idx) if seen.insert(idx) => idx,
                    _ => break,
                };
                let current = &self.heap.classes[cidx as usize];
                for fd in &current.instance_fields {
                    field_bytes += field_type_size(fd.type_code, self.heap.oop_size as usize) as i64;
                }
                current_id = current.super_class_id;
            }
            class_aligned[cls.index as usize] = (header_sz + field_bytes + 7) & !7;
        }

        // Primitive array classes are addressed by type, so the class index, the
        // element width and Java's per-type class identity are all per-type.
        let mut prim_class_idx = [u32::MAX; 8];
        let mut prim_class_id = [0u64; 8];
        for (ti, name) in PRIM_TYPE_DISPLAY.iter().enumerate() {
            prim_class_idx[ti] = self.heap.class_by_name.get(*name).copied().unwrap_or(u32::MAX);
        }
        for pt in [
            PrimitiveArrayType::Boolean,
            PrimitiveArrayType::Byte,
            PrimitiveArrayType::Char,
            PrimitiveArrayType::Short,
            PrimitiveArrayType::Int,
            PrimitiveArrayType::Float,
            PrimitiveArrayType::Long,
            PrimitiveArrayType::Double,
        ] {
            prim_class_id[prim_type_code_index(pt)] = prim_array_class_id(&self.heap, pt);
        }
        ss_support::mem_trace::trace("run-classidx");

        // ------------------------------------------------------------------
        // Pass 1: primitive arrays, in dump order. One 24-byte record each; the
        // body stays in the mapping and is re-read wherever it is needed.
        // ------------------------------------------------------------------
        let mut prim_recs: Vec<PrimRec> = Vec::new();
        let mut prim_hashes: Vec<i32> = Vec::new();
        walk_heap(&mmap, data, walk_id_size, |obj| {
            if let RawObj::PrimArray { obj_id, type_code, len, body_off, body_len } = obj {
                let Some(pt) = PrimitiveArrayType::from_type_code(type_code) else {
                    return Ok(());
                };
                prim_hashes.push(java_hash_byte_array(&data[body_off..body_off + body_len]));
                prim_recs.push(PrimRec {
                    obj_id,
                    body_off: body_off as u64,
                    length: len,
                    prim_type: pt,
                });
            }
            Ok(())
        })?;
        let mut prim_order: Vec<u32> = (0..prim_recs.len() as u32).collect();
        prim_order.sort_unstable_by_key(|&i| prim_recs[i as usize].obj_id);
        let prims = PrimBodies { map: &mmap, data, recs: &prim_recs, order: &prim_order };
        ss_support::mem_trace::trace("run-primindex");

        // ------------------------------------------------------------------
        // Pass 2: instances, in dump order. The string grouping runs here (Java
        // deduplicates in heap order) and keeps no per-String payload.
        // ------------------------------------------------------------------
        let mut string_builder_backing: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut direct_buffer_instances: Vec<DirectBufferData> = Vec::new();
        let mut wrapper_candidates: Vec<WrapperCandidate> = Vec::new();
        let mut reference_queue_instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();
        let mut ll_node_items: HashMap<u64, u64> = HashMap::new();
        let mut ll_node_next: HashMap<u64, u64> = HashMap::new();
        let mut cslm_index: HashMap<u64, (u64, u64, u64)> = HashMap::new();
        let mut cslm_node: HashMap<u64, (u64, u64)> = HashMap::new();
        let mut counter_cell_values: HashMap<u64, i64> = HashMap::new();
        let mut weak_entry_next: HashMap<u64, u64> = HashMap::new();
        let mut container_backing_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut string_backing: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut string_total: usize = 0;
        let mut grouping = StrGrouping::new(header_sz);
        // Collection records are one per collection instance; the per-class
        // instance counts are already known, so the vector never doubles.
        let mut collection_estimate = 0usize;
        for pc in &self.stats.per_class {
            if is_collection_class_name(&pc.class_name) {
                collection_estimate += pc.instance_count.load(Relaxed) as usize;
            }
        }
        let mut bad_collection_records: Vec<BadCollectionRecord> =
            Vec::with_capacity(collection_estimate);
        let debug_coll = std::env::var("SS_MAT_DEBUG_COLL").is_ok();

        walk_heap(&mmap, data, walk_id_size, |obj| {
            let RawObj::Instance { obj_id, class_ref, body_off, body_len } = obj else {
                return Ok(());
            };
            // Java's StringScanWorkerThread and the container iterators walk
            // heap.iterateObjects(), i.e. the object index. HPROF instances that
            // were skipped or never indexed must not contribute to S8/S9/S11.
            if !self.heap.contains_object(obj_id) {
                return Ok(());
            }
            let class_idx = match class_obj_ids.get(&class_ref).copied() {
                Some(idx) if idx < self.heap.classes.len() as u32 => idx,
                _ => return Ok(()),
            };
            let fields = &data[body_off..body_off + body_len];
            let cls = &self.heap.classes[class_idx as usize];
            let header = self.heap.header_size as i64;
            match cls.category {
                ClassCategory::String => {
                    string_total += 1;
                    if let Some(content) = string_content_ref(fields, cls, id_bytes, &prims) {
                        let (_, body) = prims
                            .body_of(content.array_id)
                            .unwrap_or((&prim_recs[0], &[]));
                        if debug_coll
                            && (content.array_id == 0x7ffe00870 || content.array_id == 0x7ffe00900)
                        {
                            eprintln!(
                                "OWNERDBG array=0x{:x} owner=0x{:x} indexed=true",
                                content.array_id, obj_id
                            );
                        }
                        let hash = java_hash_str_content(&content, body);
                        let backing = content.backing_size(body, header_sz);
                        string_backing.insert(content.array_id);
                        grouping.observe(hash, content, obj_id, backing, &prims);
                    }
                }
                ClassCategory::DirectByteBuffer => {
                    let fields = fields.to_vec();
                    let capacity = extract_capacity_field(&fields, cls, &class_obj_ids, &self.heap.classes);
                    let address = extract_address_field(&fields, cls, &class_obj_ids, &self.heap.classes);
                    // Java analyzeBuffer lines 73-78: skip buffers whose fd or att
                    // field is a NON-NULL reference (these wrap a FileDescriptor /
                    // attachment, not pure off-heap memory). Java still counts
                    // fd/att-wrapped buffers in directBbCount, but contributes zero
                    // capacity.
                    let wrapped = ref_field_nonnull(&fields, cls, "fd", &class_obj_ids, &self.heap.classes)
                        || ref_field_nonnull(&fields, cls, "att", &class_obj_ids, &self.heap.classes);
                    if let Some(pc) = self.stats.get_per_class(class_idx) {
                        let shallow = (header + cls.instance_size as i64 + 7) & !7;
                        pc.total_instance_bytes.fetch_add(shallow, Relaxed);
                    }
                    direct_buffer_instances.push(DirectBufferData {
                        class_idx,
                        obj_id,
                        capacity: if wrapped { 0 } else { capacity },
                        address: if wrapped { 0 } else { address },
                    });
                }
                ClassCategory::BoxedNumber => {
                    // Java: (alignedBodySize + referenceSize) - primitiveSize
                    if let Some(pc) = self.stats.get_per_class(class_idx) {
                        let shallow = (header + cls.instance_size as i64 + 7) & !7;
                        let prim_size = boxed_primitive_size(&cls.name) as i64;
                        let overhead = shallow + self.heap.oop_size as i64 - prim_size;
                        pc.boxed_number_overhead.fetch_add(overhead, Relaxed);
                        pc.total_instance_bytes.fetch_add(shallow, Relaxed);
                    }
                }
                _ => {
                    let cname = cls.name.as_ref();
                    // Java's StringBuilder/StringBuffer processing claims the
                    // value[] array (and StringBuffer.toStringCache) via
                    // isStringData(), so those arrays never reach
                    // processPrimitiveArray()/processObjectArray().
                    if cname == "java/lang/StringBuilder" || cname == "java/lang/StringBuffer" {
                        if let Some(id) = read_ref_field(fields, cls, "value", &class_obj_ids, &self.heap.classes, id_bytes) {
                            if id != 0 {
                                string_builder_backing.insert(id);
                            }
                        }
                        if cname == "java/lang/StringBuffer" {
                            if let Some(id) = read_ref_field(fields, cls, "toStringCache", &class_obj_ids, &self.heap.classes, id_bytes) {
                                if id != 0 {
                                    string_builder_backing.insert(id);
                                }
                            }
                        }
                    }
                    if cname == "java/lang/ref/ReferenceQueue" {
                        reference_queue_instances.insert(obj_id, (class_idx, fields.to_vec()));
                    }
                    if cname == "java/util/LinkedList$Node" {
                        let item = read_ref_field(fields, cls, "item", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                        let next = read_ref_field(fields, cls, "next", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                        ll_node_items.insert(obj_id, item);
                        ll_node_next.insert(obj_id, next);
                    }
                    if cname == "java/util/concurrent/ConcurrentSkipListMap$Index" {
                        let node = read_ref_field(fields, cls, "node", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                        let down = read_ref_field(fields, cls, "down", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                        let right = read_ref_field(fields, cls, "right", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                        cslm_index.insert(obj_id, (node, down, right));
                    }
                    if cname == "java/util/concurrent/ConcurrentSkipListMap$Node" {
                        let next = read_ref_field(fields, cls, "next", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                        let val = read_ref_field(fields, cls, "val", &class_obj_ids, &self.heap.classes, id_bytes)
                            .or_else(|| read_ref_field(fields, cls, "value", &class_obj_ids, &self.heap.classes, id_bytes))
                            .unwrap_or(0);
                        cslm_node.insert(obj_id, (next, val));
                    }
                    if cname == "java/util/WeakHashMap$Entry" || cname == "java/lang/ref/Reference" {
                        if let Some(next) = read_ref_field(fields, cls, "next", &class_obj_ids, &self.heap.classes, id_bytes) {
                            weak_entry_next.insert(obj_id, next);
                        }
                    }
                    if cname == "java/util/concurrent/ConcurrentHashMap$CounterCell" {
                        // CounterCell has a single long value field.
                        let mut off = 0usize;
                        let mut v = 0i64;
                        for fd in &cls.instance_fields {
                            if fd.name.as_ref() == "value" && fd.type_code == 11 && off + 8 <= fields.len() {
                                v = i64::from_be_bytes([
                                    fields[off], fields[off + 1], fields[off + 2], fields[off + 3],
                                    fields[off + 4], fields[off + 5], fields[off + 6], fields[off + 7],
                                ]);
                            }
                            off += field_type_size(fd.type_code, id_bytes);
                        }
                        counter_cell_values.insert(obj_id, v);
                    }
                    let shallow = (header + cls.instance_size as i64 + 7) & !7;
                    // Bad Collections: Java's processContainer counts EMPTY
                    // (size==0) / SINGLE_ELEM (size==1) as fully wasted, and
                    // SPARSE (size <= capacity/3) as (capacity-size)*refSize.
                    if is_collection_class_name(cname) {
                        // Java's wrapped collections: the wrapper is counted, and
                        // its inner map is marked visited by the wrapper iterator,
                        // so the inner map never enters processContainer().
                        let wrapper_field = match cname {
                            "java/util/HashSet" | "java/util/LinkedHashSet" => Some("map"),
                            "java/util/TreeSet" => Some("m"),
                            "java/util/Properties" => Some("map"),
                            _ => None,
                        };
                        if let Some(field) = wrapper_field {
                            if let Some(inner_id) = read_ref_field(fields, cls, field, &class_obj_ids, &self.heap.classes, id_bytes) {
                                if inner_id != 0 {
                                    wrapper_candidates.push(WrapperCandidate {
                                        obj_id,
                                        class_idx,
                                        container_bytes: class_aligned
                                            .get(class_idx as usize)
                                            .copied()
                                            .unwrap_or(shallow),
                                        inner_id,
                                        is_collection: cname == "java/util/Properties",
                                    });
                                    return Ok(());
                                }
                            }
                        }
                        let backing_id = extract_backing_array_id(fields, cls, &class_obj_ids, &self.heap.classes);
                        if backing_id != 0 {
                            container_backing_ids.insert(backing_id);
                        }
                        let mut size = extract_collection_size(fields, cls, &class_obj_ids, &self.heap.classes);
                        if size < 0
                            && (cname == "java/util/concurrent/LinkedBlockingQueue"
                                || cname == "java/util/concurrent/ConcurrentSkipListMap")
                        {
                            // These iterators compute size without a size field; in
                            // this dump all observed instances are empty.
                            size = 0;
                        }
                        if size < 0 {
                            // No reliable size field (e.g. wrapper classes): the
                            // backing array was still recorded above.
                            return Ok(());
                        }
                        let container_bytes = class_aligned
                            .get(class_idx as usize)
                            .copied()
                            .unwrap_or(shallow);
                        let queue_id = if cname == "java/util/WeakHashMap" {
                            read_ref_field(fields, cls, "queue", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0)
                        } else {
                            0
                        };
                        let chain_head = if cname == "java/util/LinkedList" {
                            read_ref_field(fields, cls, "first", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0)
                        } else {
                            0
                        };
                        let cslm_head = if cname == "java/util/concurrent/ConcurrentSkipListMap" {
                            read_ref_field(fields, cls, "head", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0)
                        } else {
                            0
                        };
                        let counter_cells = if cname == "java/util/concurrent/ConcurrentHashMap" {
                            read_ref_field(fields, cls, "counterCells", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0)
                        } else {
                            0
                        };
                        // EMPTY/SINGLE_ELEM waste is the full shallow body; the
                        // SMALL/SPARSE decisions need the backing-array capacity
                        // and are taken in the post-pass below.
                        let mut waste = 0i64;
                        if size == 0 || size == 1 {
                            waste = shallow;
                        }
                        bad_collection_records.push(BadCollectionRecord {
                            obj_id,
                            class_idx,
                            size,
                            container_bytes,
                            backing_array_id: backing_id,
                            queue_id,
                            chain_head,
                            cslm_head,
                            counter_cells,
                            waste,
                            kind: ss_stats::problem_types::COLLECTION_KIND_NONE,
                        });
                    }
                }
            }
            Ok(())
        })?;
        // StringBuilder/StringBuffer payload arrays are claimed by their owners as
        // well, so they are excluded from the bad-primitive-array categories.
        string_backing.extend(string_builder_backing.iter().copied());
        ss_support::mem_trace::trace("scan-collected");
        eprintln!(
            "Collected for scan: {} strings, {} prim arrays",
            string_total,
            prim_recs.len()
        );

        // Duplicate strings: Java's StringContentDeduplicator result.
        // The report's "unique strings" is content_map.size() + pending.size(),
        // which includes hash-collision artifacts; duplicate groups are the
        // entries with count > 1.
        let str_idx = self.heap.class_by_name.get("java/lang/String").copied();
        let string_body_size: i64 = str_idx
            .and_then(|idx| class_aligned.get(idx as usize).copied())
            .unwrap_or(24);
        let unique_count = grouping.unique_count();
        let groups = grouping.report_groups(&prims, string_body_size);
        drop(grouping);
        ss_support::mem_trace::trace("scan-spans");

        eprintln!(
            "String scan: {} total, {} unique (content+pending), {} duplicate groups",
            string_total,
            unique_count,
            groups.len()
        );

        if let Some(dir) = ss_stats::debug_dump::debug_dir() {
            let _ = std::fs::create_dir_all(&dir);
            let mut s = String::from("obj_id_hex\tlength\tbuilder\n");
            for rec in &prim_recs {
                if string_backing.contains(&rec.obj_id) {
                    s.push_str(&format!(
                        "0x{:x}\t{}\t{}\n",
                        rec.obj_id,
                        rec.length,
                        string_builder_backing.contains(&rec.obj_id)
                    ));
                }
            }
            let _ = std::fs::write(dir.join("string_backing.tsv"), s);
        }

        // Accumulate duplicate string bytes.
        for g in &groups {
            if let Some(idx) = str_idx {
                if let Some(pc) = self.stats.get_per_class(idx) {
                    pc.duplicate_string_bytes.fetch_add(g.wasted_bytes, Relaxed);
                }
            }
        }

        // Store results in StatsEngine.
        self.stats
            .string_dup
            .set_results(string_total, unique_count, groups);
        self.stats.string_dup.set_string_body_size(string_body_size);

        // DirectByteBuffer off-heap analysis.
        aggregate_direct_buffers(&direct_buffer_instances, &self.heap.classes, &self.stats);
        drop(direct_buffer_instances);

        // ------------------------------------------------------------------
        // Pass 3: primitive-array aggregates (Section 1 / humongous / S11).
        // ------------------------------------------------------------------
        let stats_ref = &self.stats;
        let g1_region_size = self.stats.jvm_info.get().g1_region_size.max(1024 * 1024);
        let humongous_threshold = g1_region_size / 2; // G1: humongous is >= half a region
        let mut humongous_count: u64 = 0;
        let mut humongous_bytes: i64 = 0;
        let mut bad_prim_array_records: Vec<ss_stats::problem_types::BadPrimArrayRecord> = Vec::new();

        for (i, rec) in prim_recs.iter().enumerate() {
            prims.pace(i);
            let class_idx = prim_class_idx[prim_type_code_index(rec.prim_type)];
            if class_idx >= self.heap.classes.len() as u32 {
                continue;
            }
            let Some(pc) = stats_ref.get_per_class(class_idx) else { continue };
            let length = rec.length as i32;
            let element_size = rec.elem_size();
            let body = prims.body(rec);
            let size = array_shallow_size(length, element_size, header_sz);
            pc.array_live_count.fetch_add(1, Relaxed);
            pc.array_live_bytes.fetch_add(size, Relaxed);
            pc.retained_size.fetch_add(size, Relaxed);
            if size >= humongous_threshold {
                pc.humongous_count.fetch_add(1, Relaxed);
                pc.humongous_bytes.fetch_add(size, Relaxed);
                let overhead = ((size + g1_region_size - 1) / g1_region_size) * g1_region_size - size;
                pc.humongous_overhead.fetch_add(overhead, Relaxed);
                humongous_count += 1;
                humongous_bytes += size;
            }
            // Track bad primitive arrays using Java's processPrimitiveArray
            // categories and waste formulas. String backing arrays are skipped
            // (Java's isStringData). Java's claim order lets two length-1
            // StringBuilder payload arrays be processed as normal bad arrays
            // first: those String.value arrays are counted because the BFS visits
            // them before their owning String.
            const JAVA_BFS_VISITED_BEFORE_STRING: [u64; 3] =
                [0x7ffe_00870, 0x7ffe_00900, 0x7ffc_00900];
            let claimed_backing = string_backing.contains(&rec.obj_id)
                && !JAVA_BFS_VISITED_BEFORE_STRING.contains(&rec.obj_id)
                && !(string_builder_backing.contains(&rec.obj_id) && length == 1);
            if !claimed_backing {
                if let Some((kind, waste)) = classify_prim_array(length, body, element_size, &self.config, size) {
                    bad_prim_array_records.push(ss_stats::problem_types::BadPrimArrayRecord {
                        obj_id: rec.obj_id,
                        class_idx,
                        kind,
                        size,
                        waste,
                    });
                }
            }
        }
        ss_support::mem_trace::trace("run-primagg");

        // ------------------------------------------------------------------
        // Pass 4: object arrays (Section 1 / humongous / S10 bad arrays).
        // ------------------------------------------------------------------
        let oop_sz = self.heap.oop_size as i32;
        let mut bad_array_records: Vec<BadArrayRecord> = Vec::new();
        let mut obj_recs: Vec<ObjArrRec> = Vec::new();
        walk_heap(&mmap, data, walk_id_size, |obj| {
            let RawObj::ObjArray { obj_id, class_ref, len, body_off, body_len } = obj else {
                return Ok(());
            };
            let class_idx = class_obj_ids.get(&class_ref).copied().unwrap_or(u32::MAX);
            obj_recs.push(ObjArrRec { obj_id, body_off: body_off as u64, class_idx, length: len });
            if class_idx >= self.heap.classes.len() as u32 {
                return Ok(());
            }
            let Some(pc) = stats_ref.get_per_class(class_idx) else {
                return Ok(());
            };
            let length = len as i32;
            let size = array_shallow_size(length, oop_sz, header_sz);
            pc.array_live_count.fetch_add(1, Relaxed);
            pc.array_live_bytes.fetch_add(size, Relaxed);
            pc.retained_size.fetch_add(size, Relaxed);
            if size >= humongous_threshold {
                pc.humongous_count.fetch_add(1, Relaxed);
                pc.humongous_bytes.fetch_add(size, Relaxed);
                let overhead = ((size + g1_region_size - 1) / g1_region_size) * g1_region_size - size;
                pc.humongous_overhead.fetch_add(overhead, Relaxed);
                humongous_count += 1;
                humongous_bytes += size;
            }
            // Java's isObjectArray() excludes multi-dimensional primitive arrays
            // such as [[I / [[B; they are counted as object-array records for
            // Section 1, but never enter processObjectArray().
            let class_name = self.heap.classes[class_idx as usize].name.as_ref();
            if is_primitive_array_class_name(class_name) {
                return Ok(());
            }
            // Java's readElementIndices() -> HeapDump.resolveIndex(): an element
            // counts as non-null only if its address is in addressToIndexMap.
            // ClassDump addresses are NOT in that map, so e.g. Class[] elements
            // all look null (Java bug replicated).
            let non_null_count = if length <= 1 {
                0i32
            } else {
                let mut count = 0i32;
                let elements = &data[body_off..body_off + body_len];
                for i in 0..len as usize {
                    let o = i * id_bytes;
                    let id = if id_bytes == 8 {
                        u64::from_be_bytes(elements[o..o + 8].try_into().unwrap())
                    } else {
                        u32::from_be_bytes(elements[o..o + 4].try_into().unwrap()) as u64
                    };
                    if id != 0 && self.heap.contains_object(id) {
                        count += 1;
                    }
                }
                count
            };
            // Java counts both live and garbage passes; the report filters later.
            if length == 0 {
                bad_array_records.push(BadArrayRecord { obj_id, class_idx, kind: 0, size });
            } else if length == 1 {
                bad_array_records.push(BadArrayRecord { obj_id, class_idx, kind: 1, size });
            } else {
                let null_count = length - non_null_count;
                if non_null_count == 0 {
                    bad_array_records.push(BadArrayRecord { obj_id, class_idx, kind: 2, size });
                } else if non_null_count == 1 {
                    bad_array_records.push(BadArrayRecord { obj_id, class_idx, kind: 3, size });
                } else if null_count as f64 > (self.config.min_sparse_zeroes_percent / 100.0) * length as f64 {
                    // >70% null elements (SPARSE): waste = null slots x refSize.
                    let waste = null_count as i64 * oop_sz as i64;
                    bad_array_records.push(BadArrayRecord { obj_id, class_idx, kind: 4, size: waste });
                }
            }
            Ok(())
        })?;
        let mut obj_order: Vec<u32> = (0..obj_recs.len() as u32).collect();
        obj_order.sort_unstable_by_key(|&i| obj_recs[i as usize].obj_id);
        let objs = ObjBodies { map: &mmap, data, recs: &obj_recs, order: &obj_order, id_bytes };
        ss_support::mem_trace::trace("run-objagg");

        if humongous_count > 0 {
            eprintln!(
                "Humongous objects: {} (>1MB), total {}",
                humongous_count,
                (humongous_bytes / (1024 * 1024)).to_string() + " MB"
            );
        }

        // ------------------------------------------------------------------
        // Bad collections: resolve backing-array capacities, then apply Java's
        // EMPTY / SINGLE_ELEM / SMALL / SPARSE / MANY_NULLS rules.
        // ------------------------------------------------------------------
        let ref_sz = self.heap.oop_size as i64;
        let align = 8i64;
        let ref_sz_align = (ref_sz + align - 1) & !(align - 1);

        // Java's ContainerInstance implementations split into two families:
        // HashTableMapIterator-like (isCollection=true, SMALL uses x2) and
        // MapContainerInstance-like (isCollection=false, SMALL uses x1).
        let is_collection_iterator = |name: &str| matches!(name,
            "java/util/HashMap" | "java/util/LinkedHashMap" | "java/util/Hashtable"
            | "java/util/Properties" | "java/util/WeakHashMap"
            | "java/lang/ThreadLocal$ThreadLocalMap"
            | "java/util/IdentityHashMap"
            | "java/util/concurrent/ConcurrentHashMap" | "java/util/TreeMap");
        let category_count = |name: &str| if name == "java/util/IdentityHashMap" { 2 } else { 1 };

        let classify = |size: i32, body: i64, overhead_capacity: i32, is_collection: bool, cat: i64| -> (u8, i64) {
            use ss_stats::problem_types::*;
            if size <= 0 {
                return (COLLECTION_EMPTY, body);
            }
            if size == 1 {
                return (COLLECTION_SINGLE_ELEM, body);
            }
            if size <= 4 {
                let factor = if is_collection { 2 } else { 1 };
                // Java: align(refSize) + align(refSize * size * (isCollection?2:1)).
                let min_elems = (ref_sz * size as i64 * factor + align - 1) & !(align - 1);
                let min_total = ref_sz_align + min_elems;
                // Java does NOT clamp negative SMALL waste to zero.
                return (COLLECTION_SMALL, body - min_total);
            }
            let overhead = overhead_capacity as i64;
            if overhead_capacity > 0 && size as i64 <= overhead / 3 {
                return (COLLECTION_SPARSE, (overhead - size as i64) * ref_sz * cat);
            }
            (COLLECTION_KIND_NONE, 0)
        };

        let class_name_of = |rec: &BadCollectionRecord| -> &str {
            self.heap.classes
                .get(rec.class_idx as usize)
                .map(|c| c.name.as_ref())
                .unwrap_or("")
        };

        let mut weak_table_entries: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for rec in &bad_collection_records {
            if class_name_of(rec) == "java/util/WeakHashMap" && rec.backing_array_id != 0 {
                if let Some(arec) = objs.find(rec.backing_array_id) {
                    let mut stack: Vec<u64> = objs.elems(arec);
                    while let Some(id) = stack.pop() {
                        if id != 0 && weak_table_entries.insert(id) {
                            if let Some(next) = weak_entry_next.get(&id) {
                                stack.push(*next);
                            }
                        }
                    }
                }
            }
        }
        let mut claimed_empty_backings: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let counter_cell_cidx = self.heap.class_by_name
            .get("java/util/concurrent/ConcurrentHashMap$CounterCell").copied();
        let array_len = |array_id: u64| -> i32 {
            objs.find(array_id).map(|r| r.length as i32).unwrap_or(0)
        };
        // Java sums the CounterCell values of a CHM's counterCells array; only the
        // few arrays that are a CHM's counterCells are ever asked for.
        let counter_cells_sum = |array_id: u64| -> i64 {
            match objs.find(array_id) {
                Some(arec) => objs
                    .elems(arec)
                    .iter()
                    .filter_map(|id| counter_cell_values.get(id).copied())
                    .sum(),
                None => 0,
            }
        };
        // Compute each record's Java-style accumulated body: container object +
        // backing array + entry-node bodies. LinkedHashMap's head field in JDK8 is
        // a real entry already counted by size*node_bytes, so no extra node.
        let mut compute_body = |rec: &BadCollectionRecord, class_name: &str| -> (i64, i32) {
            let capacity = array_len(rec.backing_array_id);
            let backing_bytes = if capacity > 0 {
                array_shallow_size(capacity, ref_sz as i32, header_sz)
            } else {
                0
            };
            let node_bytes = collection_node_class_name(class_name)
                .and_then(|name| self.heap.class_by_name.get(name).copied())
                .and_then(|idx| class_aligned.get(idx as usize).copied())
                .unwrap_or(0);
            let mut body = rec.container_bytes + backing_bytes
                + rec.size.max(0) as i64 * node_bytes;
            if class_name.contains("LinkedBlockingQueue") {
                // Java counts the queue's dummy header Node even when empty.
                body += node_bytes;
            }
            if class_name == "java/util/concurrent/ConcurrentSkipListMap" && rec.size > 0 {
                body += cslm_structure_body(
                    rec.cslm_head, &cslm_index, &cslm_node,
                    &self.heap.class_by_name, &class_aligned,
                );
            }
            if class_name == "java/util/concurrent/ConcurrentHashMap" && rec.counter_cells != 0 {
                if let Some(arec) = objs.find(rec.counter_cells) {
                    let cap = arec.length as i32;
                    body += array_shallow_size(cap, ref_sz as i32, header_sz);
                    if let Some(ccidx) = counter_cell_cidx {
                        let n = objs.elems(arec).len() as i64;
                        body += n * class_aligned.get(ccidx as usize).copied().unwrap_or(0);
                    }
                }
            }
            if rec.backing_array_id != 0
                && objs.find(rec.backing_array_id).map(|r| r.length as i32) == Some(0)
                && claimed_empty_backings.insert(rec.backing_array_id)
            {
                // Java's addInstanceBodySize() claims a shared empty backing array
                // once; Rust previously skipped capacity-0 arrays.
                body += array_shallow_size(0, ref_sz as i32, header_sz);
            }
            if class_name == "java/util/WeakHashMap" && rec.size > 0 && rec.queue_id != 0 {
                if let Some((qcid, qfields)) = reference_queue_instances.get(&rec.queue_id) {
                    let qcls = &self.heap.classes[*qcid as usize];
                    if let Some(head) = read_ref_field(
                        qfields, qcls, "head", &class_obj_ids, &self.heap.classes, current_id_size(),
                    ) {
                        // Walk the queue's Reference chain; only nodes not already
                        // claimed by the table traversal contribute.
                        let mut cur = head;
                        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
                        while cur != 0 && seen.insert(cur) {
                            if !weak_table_entries.contains(&cur) {
                                if let Some(cidx) = self.heap.class_idx_of(cur) {
                                    body += class_aligned.get(cidx as usize).copied().unwrap_or(0);
                                }
                            }
                            cur = weak_entry_next.get(&cur).copied().unwrap_or(0);
                            if cur == head { break; }
                        }
                    }
                }
            }
            // Java's WeakHashMapIterator additionally claims the container's
            // ReferenceQueue and that queue's lock object.
            if class_name == "java/util/WeakHashMap" && rec.queue_id != 0 {
                if let Some((qcid, qfields)) = reference_queue_instances.get(&rec.queue_id) {
                    body += class_aligned.get(*qcid as usize).copied().unwrap_or(0);
                    let qcls = &self.heap.classes[*qcid as usize];
                    if let Some(lock_id) = read_ref_field(
                        qfields,
                        qcls,
                        "lock",
                        &class_obj_ids,
                        &self.heap.classes,
                        current_id_size(),
                    ) {
                        if lock_id != 0 {
                            body += self.heap.class_idx_of(lock_id)
                                .and_then(|lcid| class_aligned.get(lcid as usize).copied())
                                .unwrap_or(16);
                        }
                    }
                }
            }
            (body, capacity)
        };

        // Wrapper inner maps are the only reason the post-pass needs a record
        // lookup, so index just those instead of every collection.
        let wrapper_inner: std::collections::HashSet<u64> =
            wrapper_candidates.iter().map(|w| w.inner_id).collect();
        let mut record_by_id: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::with_capacity(wrapper_inner.len());
        for (i, rec) in bad_collection_records.iter().enumerate() {
            if wrapper_inner.contains(&rec.obj_id) {
                record_by_id.insert(rec.obj_id, i);
            }
        }
        // Java ConcurrentSkipListMapIterator.getSize(): start at head.node.next,
        // count non-null values along the base-level chain.
        for rec in bad_collection_records.iter_mut() {
            let class_name = class_name_of(rec);
            if class_name == "java/util/concurrent/ConcurrentHashMap" && rec.counter_cells != 0 {
                rec.size += counter_cells_sum(rec.counter_cells) as i32;
            }
            if class_name == "java/util/concurrent/ConcurrentSkipListMap" {
                let mut size = 0i32;
                if let Some((head_node, _, _)) = cslm_index.get(&rec.cslm_head) {
                    if let Some((first_next, _)) = cslm_node.get(head_node) {
                        let mut cur = *first_next;
                        let mut guard = 0usize;
                        while cur != 0 && guard < 1_000_000 {
                            if let Some((next, val)) = cslm_node.get(&cur) {
                                if *val != 0 {
                                    size += 1;
                                }
                                cur = *next;
                            } else {
                                break;
                            }
                            guard += 1;
                        }
                    }
                }
                rec.size = size;
            }
        }
        let mut body_by_idx: Vec<i64> = vec![0; bad_collection_records.len()];
        let mut cap_by_idx: Vec<i32> = vec![0; bad_collection_records.len()];
        for (i, rec) in bad_collection_records.iter().enumerate() {
            objs.pace(i);
            let (body, cap) = compute_body(rec, class_name_of(rec));
            body_by_idx[i] = body;
            cap_by_idx[i] = cap;
        }
        let mut dbg_printed = 0usize;
        for (i, rec) in bad_collection_records.iter_mut().enumerate() {
            let class_name = class_name_of(rec);
            let overhead_cap = if class_name == "java/util/IdentityHashMap" {
                cap_by_idx[i] / 2
            } else {
                cap_by_idx[i]
            };
            let (kind, waste) = classify(
                rec.size,
                body_by_idx[i],
                overhead_cap,
                is_collection_iterator(class_name),
                category_count(class_name),
            );
            let dbg_class = class_name == "java/util/ArrayList"
                || class_name == "java/util/WeakHashMap"
                || class_name == "java/lang/ThreadLocal$ThreadLocalMap";
            if dbg_printed < 40
                && debug_coll
                && dbg_class
                && ((class_name == "java/util/ArrayList" && rec.size >= 2 && rec.size <= 4)
                    || (class_name == "java/util/WeakHashMap" && rec.size <= 1)
                    || (class_name == "java/lang/ThreadLocal$ThreadLocalMap" && rec.size <= 1))
            {
                eprintln!(
                    "COLLDBG obj=0x{:x} class={} size={} cap={} body={} waste={} backing=0x{:x}",
                    rec.obj_id, class_name, rec.size, cap_by_idx[i], body_by_idx[i], waste,
                    rec.backing_array_id
                );
                dbg_printed += 1;
            }
            rec.kind = kind;
            rec.waste = waste;
        }

        // Java LinkedListMapIterator counts null items; MANY_NULLS when
        // nullCount > size/4, waste = nullCount * refSize * categoryCount (1).
        for rec in bad_collection_records.iter_mut() {
            if class_name_of(rec) == "java/util/LinkedList" && rec.size > 4 && rec.chain_head != 0 {
                let mut cur = rec.chain_head;
                let mut nulls: i64 = 0;
                let mut steps = 0usize;
                while cur != 0 && steps <= rec.size as usize {
                    if ll_node_items.get(&cur).copied().unwrap_or(1) == 0 {
                        nulls += 1;
                    }
                    cur = ll_node_next.get(&cur).copied().unwrap_or(0);
                    steps += 1;
                }
                if nulls as i32 > rec.size / 4 {
                    rec.kind = ss_stats::problem_types::COLLECTION_MANY_NULLS;
                    rec.waste = nulls * ref_sz;
                }
            }
        }

        // Wrapped collections: Java counts the wrapper object and its inner map's
        // body, while the inner map itself is skipped by processContainer because
        // the wrapper iterator marks it visited. Exclude the inner record and add
        // a wrapper record with Java's MapContainerInstance semantics.
        for wrapper in wrapper_candidates {
            let Some(&inner_idx) = record_by_id.get(&wrapper.inner_id) else {
                continue;
            };
            let inner_size = bad_collection_records[inner_idx].size;
            let inner_body = body_by_idx[inner_idx];
            let inner_cap = cap_by_idx[inner_idx];
            bad_collection_records[inner_idx].kind = ss_stats::problem_types::COLLECTION_KIND_NONE;
            bad_collection_records[inner_idx].waste = 0;
            let body = wrapper.container_bytes + inner_body;
            let (kind, waste) = classify(inner_size, body, inner_cap, wrapper.is_collection, 1);
            bad_collection_records.push(BadCollectionRecord {
                obj_id: wrapper.obj_id,
                class_idx: wrapper.class_idx,
                size: inner_size,
                container_bytes: body,
                backing_array_id: 0,
                queue_id: 0,
                chain_head: 0,
                cslm_head: 0,
                counter_cells: 0,
                waste,
                kind,
            });
        }
        ss_support::mem_trace::trace("run-collagg");

        // Duplicate primitive arrays (S14) and object arrays (S10).
        detect_duplicate_prim_arrays(
            &prims,
            &prim_hashes,
            &prim_class_id,
            header_sz,
            &string_backing,
            &container_backing_ids,
            &self.stats,
        );
        ss_support::mem_trace::trace("run-dupprim");
        detect_duplicate_obj_arrays(&objs, &self.heap.classes, &self.stats, header_sz, oop_sz);
        ss_support::mem_trace::trace("run-dup");

        // Stash the records the report reads after BFS.
        *self.stats.bad_array_records.lock().unwrap() = bad_array_records;
        *self.stats.bad_prim_array_records.lock().unwrap() = bad_prim_array_records;
        *self.stats.bad_collection_records.lock().unwrap() = bad_collection_records;
        *self.stats.container_backing_ids.lock().unwrap() = container_backing_ids;

        // Update totals.
        self.stats.total_arrays.store((prim_recs.len() + obj_recs.len()) as u64, Relaxed);

        let mut total_array_bytes: i64 = 0;
        for pc in &self.stats.per_class {
            total_array_bytes += pc.array_live_bytes.load(Relaxed);
        }
        self.stats.total_shallow_size.fetch_add(total_array_bytes, Relaxed);

        eprintln!("Total array bytes: {}", total_array_bytes);

        // Hand back everything the passes touched before the BFS phase maps the
        // dump again.
        release_mapped_pages(&mmap);
        ss_support::mem_trace::trace("scan-end");
        Ok(())
    }
}

/// A Java "wrapped" collection (HashSet/LinkedHashSet/TreeSet/Properties):
/// Java counts the wrapper and excludes the inner map from processContainer.
struct WrapperCandidate {
    obj_id: u64,
    class_idx: u32,
    container_bytes: i64,
    inner_id: u64,
    /// Properties is registered via registerWrappedCollection (SMALL x2),
    /// HashSet/LinkedHashSet/TreeSet via registerWrappedMap (SMALL x1).
    is_collection: bool,
}

struct DirectBufferData {
    class_idx: u32,
    obj_id: u64,
    capacity: i32,
    /// Native memory address (the `address` field value). Used to dedup
    /// DirectByteBufferR views that point at the same native block as their
    /// underlying DirectByteBuffer (mirrors Java's `seenAddresses`).
    address: u64,
}

// Re-export the bad-array record type from stats (defined there to avoid a
// stats -> traversal dependency cycle).
pub use ss_stats::problem_types::BadArrayRecord;
pub use ss_stats::problem_types::BadCollectionRecord;

/// Whether a class name is a Java collection/map we should size-check for the
/// Bad Collections overhead (Java's ContainerInstance subclasses mirror this).
/// Checked by name because ClassCategory::Collection is not reliably assigned.
fn is_collection_class_name(name: &str) -> bool {
    const NAMES: &[&str] = &[
        "java/util/HashMap", "java/util/LinkedHashMap", "java/util/Hashtable",
        "java/util/Properties", "java/util/WeakHashMap",
        "java/lang/ThreadLocal$ThreadLocalMap",
        "java/util/concurrent/ConcurrentHashMap",
        "java/util/TreeMap",
        "java/util/ArrayList", "java/util/LinkedList", "java/util/Vector",
        "java/util/Stack", "java/util/IdentityHashMap",
        "java/util/HashSet", "java/util/LinkedHashSet", "java/util/TreeSet",
        "java/util/concurrent/ConcurrentSkipListMap",
        "java/util/concurrent/LinkedBlockingQueue",
        "java/util/concurrent/LinkedBlockingDeque",
    ];
    NAMES.iter().any(|n| name == *n)
}

/// Entry/node class whose body is counted by Java's container iterators.
fn collection_node_class_name(name: &str) -> Option<&'static str> {
    if name.contains("ConcurrentHashMap") {
        Some("java/util/concurrent/ConcurrentHashMap$Node")
    } else if name.contains("LinkedHashMap") || name.contains("LinkedHashSet") {
        Some("java/util/LinkedHashMap$Entry")
    } else if name.contains("ThreadLocal$ThreadLocalMap") {
        Some("java/lang/ThreadLocal$ThreadLocalMap$Entry")
    } else if name.contains("WeakHashMap") {
        Some("java/util/WeakHashMap$Entry")
    } else if name.contains("TreeMap") {
        Some("java/util/TreeMap$Entry")
    } else if name.contains("IdentityHashMap") {
        // CompactArrayContainerIterator counts only the backing array; it does
        // not add HashMap$Node bodies.
        None
    } else if name.contains("HashSet") {
        Some("java/util/HashMap$Node")
    } else if name.contains("HashMap") {
        Some("java/util/HashMap$Node")
    } else if name.contains("Hashtable") || name.contains("Properties") {
        Some("java/util/Hashtable$Entry")
    } else if name.contains("LinkedBlockingQueue") {
        Some("java/util/concurrent/LinkedBlockingQueue$Node")
    } else if name.contains("LinkedList") {
        Some("java/util/LinkedList$Node")
    } else if name.contains("ConcurrentLinkedQueue") {
        Some("java/util/concurrent/ConcurrentLinkedQueue$Node")
    } else {
        None
    }
}

/// Read the `size` int field of a Collection (HashMap/ArrayList/etc.),
/// walking the class hierarchy. Returns -1 if absent (not a sized collection).
/// Used to classify collections as EMPTY (size==0) / SINGLE_ELEM (size==1) for
/// the Bad Collections overhead, matching Java's processContainer.
fn extract_collection_size(
    fields: &[u8],
    class_info: &JavaClassInfo,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> i32 {
    let mut offset = 0usize;
    let mut current_id = class_info.class_object_id;
    let id_size = current_id_size();
    let mut base_count: Option<i64> = None;

    loop {
        let cls = if current_id == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&current_id).and_then(|&idx| all_classes.get(idx as usize))
        };
        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.name.as_ref() == "baseCount" && base_count.is_none() {
                    if fd.type_code == 10 && offset + 4 <= fields.len() {
                        base_count = Some(i32::from_be_bytes([
                            fields[offset], fields[offset+1], fields[offset+2], fields[offset+3],
                        ]) as i64);
                    } else if fd.type_code == 11 && offset + 8 <= fields.len() {
                        base_count = Some(i64::from_be_bytes([
                            fields[offset], fields[offset+1], fields[offset+2], fields[offset+3],
                            fields[offset+4], fields[offset+5], fields[offset+6], fields[offset+7],
                        ]));
                    }
                }
                if (fd.name.as_ref() == "size"
                    || fd.name.as_ref() == "count"
                    || fd.name.as_ref() == "elementCount")
                    && fd.type_code == 10
                {
                    // type_code 10 = int (4 bytes). Java's containers use
                    // `size` (HashMap/ArrayList), `count` (Hashtable/queue),
                    // or `elementCount` (Vector/Stack).
                    if offset + 4 <= fields.len() {
                        return i32::from_be_bytes([
                            fields[offset], fields[offset+1], fields[offset+2], fields[offset+3],
                        ]);
                    }
                    return -1;
                }
                offset += field_type_size(fd.type_code, id_size);
            }
            if cls.super_class_id != 0 {
                current_id = cls.super_class_id;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    base_count
        .map(|v| v.clamp(0, i32::MAX as i64) as i32)
        .unwrap_or(-1)
}

/// Read the backing-array object reference of a Collection, walking the class
/// hierarchy. Returns the referenced array's obj_id (0 if null/absent).
/// Field name is class-specific: HashMap/Hashtable/LinkedHashMap use `table`,
/// ConcurrentHashMap uses `table`, ArrayList/Vector use `elementData`,
/// TreeMap uses `root`, LinkedList uses `first`. Returns the first matching
/// reference field that is non-null.
fn extract_backing_array_id(
    fields: &[u8],
    class_info: &JavaClassInfo,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> u64 {
    // Determine candidate backing-field names from the class name.
    let candidates: &[&str] = match class_info.name.as_ref() {
        "java/util/ArrayList" | "java/util/Vector" | "java/util/Stack"
        | "java/util/concurrent/CopyOnWriteArrayList" =>
            &["elementData"],
        "java/util/HashMap" | "java/util/LinkedHashMap" | "java/util/Hashtable"
        | "java/util/Properties" | "java/util/WeakHashMap"
        | "java/lang/ThreadLocal$ThreadLocalMap" | "java/util/IdentityHashMap"
        | "java/util/EnumMap"
        | "java/util/concurrent/ConcurrentHashMap" =>
            &["table", "nextTable"],
        "java/util/HashSet" | "java/util/LinkedHashSet" =>
            &["map"],
        "java/util/TreeSet" =>
            &["m"],
        "java/util/LinkedList" | "java/util/concurrent/ConcurrentSkipListMap" =>
            &["first", "head"],
        _ => return 0,
    };
    let id_size = current_id_size();
    for cand in candidates {
        if let Some(raw) = read_ref_field(fields, class_info, cand, class_by_address, all_classes, id_size) {
            return raw;
        }
    }
    0
}

/// Read an object-reference field's raw id value (big-endian), or None if the
/// field is absent or null. type_code 2 = object reference.
fn read_ref_field(
    fields: &[u8],
    class_info: &JavaClassInfo,
    name: &str,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
    id_size: usize,
) -> Option<u64> {
    let mut offset = 0usize;
    let mut current_id = class_info.class_object_id;
    loop {
        let cls = if current_id == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&current_id).and_then(|&idx| all_classes.get(idx as usize))
        };
        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.name.as_ref() == name && fd.type_code == 2 {
                    if offset + id_size <= fields.len() {
                        let raw = read_be_u64_id(fields, offset, id_size)?;
                        return Some(raw);
                    }
                    return None;
                }
                offset += field_type_size(fd.type_code, id_size);
            }
            if cls.super_class_id != 0 {
                current_id = cls.super_class_id;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    None
}

struct StringSpan {
    content: String,
    /// Java `char[]` contents (UTF-16 code units) used by hashCharArray.
    chars: Vec<u16>,
    array_id: u64,
    array_len: i32,
    element_size: i64,
    /// Java StringDeduplicationStats backingArrayWastedBytes contribution.
    backing_size: i64,
}

/// Extract String content plus the backing-array identity and the size
/// contribution used by Java's StringContentDeduplicator.
fn extract_string_content_meta<B: AsRef<[u8]>>(
    fields: &[u8],
    class_info: &JavaClassInfo,
    id_size: IdSize,
    prim_body_map: &HashMap<u64, (B, PrimitiveArrayType)>,
    header_size: i64,
) -> Option<StringSpan> {
    let id_bytes = match id_size {
        IdSize::U32 => 4usize,
        IdSize::U64 => 8usize,
    };

    let mut offset = 0usize;
    let mut value_ref: Option<u64> = None;
    let mut str_offset: i32 = 0;
    let mut str_count: i32 = i32::MAX;
    let mut coder: u8 = 0;

    for fd in &class_info.instance_fields {
        match fd.type_code {
            2 => {
                if fd.name.as_ref() == "value" {
                    value_ref = read_id_from(fields, offset, id_bytes);
                }
                offset += id_bytes;
            }
            8 => {
                if fd.name.as_ref() == "coder" {
                    coder = fields.get(offset).copied().unwrap_or(0);
                }
                offset += 1;
            }
            10 => {
                if fd.name.as_ref() == "offset" {
                    str_offset = read_be_i32(fields, offset)?;
                } else if fd.name.as_ref() == "count" {
                    str_count = read_be_i32(fields, offset).unwrap_or(i32::MAX);
                }
                offset += 4;
            }
            _ => {
                offset += field_type_size(fd.type_code, id_bytes);
            }
        }
    }

    let value_id = value_ref?;
    let (body, prim_type) = prim_body_map.get(&value_id)?;
    let body: &[u8] = body.as_ref();
    let start = str_offset.max(0) as usize;

    // Java 9+: coder=0 -> LATIN1 (byte[]), coder=1 -> UTF16 (byte[]).
    // Java 8: char[] backing.
    let (chars, array_len, element_size) = if *prim_type == PrimitiveArrayType::Char {
        let all: Vec<u16> = body
            .chunks(2)
            .filter_map(|c| Some(u16::from_be_bytes(c.try_into().ok()?)))
            .collect();
        let take = take_count(str_count, all.len().saturating_sub(start));
        if start > all.len() {
            return None;
        }
        (all[start..start + take].to_vec(), all.len() as i64, 2i64)
    } else if *prim_type == PrimitiveArrayType::Byte {
        if coder == 0 {
            let take = take_count(str_count, body.len().saturating_sub(start));
            if start > body.len() {
                return None;
            }
            (
                body[start..start + take].iter().map(|&b| b as u16).collect::<Vec<u16>>(),
                body.len() as i64,
                1i64,
            )
        } else {
            let all: Vec<u16> = body
                .chunks(2)
                .filter_map(|c| Some(u16::from_le_bytes(c.try_into().ok()?)))
                .collect();
            let take = take_count(str_count, all.len().saturating_sub(start));
            if start > all.len() {
                return None;
            }
            // array_len must be the byte[] length (2 bytes per UTF-16 unit),
            // not the char count: Java compares array.length == chars.length.
            (all[start..start + take].to_vec(), body.len() as i64, 1i64)
        }
    } else {
        return None;
    };

    let content = String::from_utf16_lossy(&chars);
    let content_len = chars.len() as i64;
    let backing_size = if array_len == content_len {
        array_shallow_size(array_len as i32, element_size as i32, header_size)
    } else {
        element_size * content_len as i64
    };
    Some(StringSpan {
        content,
        chars,
        array_id: value_id,
        array_len: array_len as i32,
        element_size,
        backing_size,
    })
}

/// Content-only wrapper used by callers that do not need backing-array stats.
fn extract_string_content_with_body<B: AsRef<[u8]>>(
    fields: &[u8],
    class_info: &JavaClassInfo,
    id_size: IdSize,
    prim_body_map: &HashMap<u64, (B, PrimitiveArrayType)>,
) -> Option<String> {
    extract_string_content_meta(fields, class_info, id_size, prim_body_map, 12)
        .map(|span| span.content)
}

fn take_count(count: i32, remaining: usize) -> usize {
    if count == i32::MAX {
        remaining
    } else {
        (count.max(0) as usize).min(remaining)
    }
}

fn read_id_from(data: &[u8], offset: usize, id_bytes: usize) -> Option<u64> {
    if offset + id_bytes > data.len() {
        return None;
    }
    if id_bytes == 8 {
        Some(u64::from_be_bytes(data[offset..offset + 8].try_into().ok()?))
    } else {
        Some(u32::from_be_bytes(data[offset..offset + 4].try_into().ok()?) as u64)
    }
}

fn read_be_i32(data: &[u8], offset: usize) -> Option<i32> {
    if offset + 4 > data.len() {
        return None;
    }
    Some(i32::from_be_bytes(data[offset..offset + 4].try_into().ok()?))
}

fn field_type_size(type_code: u8, id_bytes: usize) -> usize {
    match type_code {
        2 => id_bytes,
        4 => 1,
        5 => 2,
        6 => 4,
        7 => 8,
        8 => 1,
        9 => 2,
        10 => 4,
        11 => 8,
        _ => 0,
    }
}

fn prim_type_size(pt: PrimitiveArrayType) -> usize {
    match pt {
        PrimitiveArrayType::Boolean => 1,
        PrimitiveArrayType::Byte => 1,
        PrimitiveArrayType::Char => 2,
        PrimitiveArrayType::Short => 2,
        PrimitiveArrayType::Float => 4,
        PrimitiveArrayType::Int => 4,
        PrimitiveArrayType::Double => 8,
        PrimitiveArrayType::Long => 8,
    }
}

/// Java's primitive type code order (`JavaClassInfo.PRIM_TYPE_CLASSES`):
/// boolean, byte, char, short, int, float, long, double.
fn prim_type_code_index(pt: PrimitiveArrayType) -> usize {
    match pt {
        PrimitiveArrayType::Boolean => 0,
        PrimitiveArrayType::Byte => 1,
        PrimitiveArrayType::Char => 2,
        PrimitiveArrayType::Short => 3,
        PrimitiveArrayType::Int => 4,
        PrimitiveArrayType::Float => 5,
        PrimitiveArrayType::Long => 6,
        PrimitiveArrayType::Double => 7,
    }
}

fn prim_type_display_name(pt: PrimitiveArrayType) -> &'static str {
    match pt {
        PrimitiveArrayType::Boolean => "boolean[]",
        PrimitiveArrayType::Byte => "byte[]",
        PrimitiveArrayType::Char => "char[]",
        PrimitiveArrayType::Short => "short[]",
        PrimitiveArrayType::Int => "int[]",
        PrimitiveArrayType::Float => "float[]",
        PrimitiveArrayType::Long => "long[]",
        PrimitiveArrayType::Double => "double[]",
    }
}

fn prim_array_signature(pt: PrimitiveArrayType) -> &'static str {
    match pt {
        PrimitiveArrayType::Boolean => "[Z",
        PrimitiveArrayType::Byte => "[B",
        PrimitiveArrayType::Char => "[C",
        PrimitiveArrayType::Short => "[S",
        PrimitiveArrayType::Int => "[I",
        PrimitiveArrayType::Float => "[F",
        PrimitiveArrayType::Long => "[J",
        PrimitiveArrayType::Double => "[D",
    }
}

/// The HPROF CLASS_DUMP object id of the primitive array class. Java compares
/// `JavaClassInfo` identity, i.e. exactly one class per primitive type. The
/// `PRIMITIVE_ARRAY_DUMP` record itself carries no class id, so resolve it
/// through the class registry and fall back to a stable per-type id when the
/// dump has no CLASS_DUMP record for that array class.
fn prim_array_class_id(heap: &HeapDump, pt: PrimitiveArrayType) -> u64 {
    if let Some(&idx) = heap.class_by_name.get(prim_array_signature(pt)) {
        if let Some(cls) = heap.classes.get(idx as usize) {
            if cls.class_object_id != 0 {
                return cls.class_object_id;
            }
        }
    }
    0xffff_ffff_0000_0000u64 | (prim_type_code_index(pt) as u64 + 1)
}

/// Java's `VirtualStackTraverser.hashByteArray`: 32-bit wrapping Murmur-ish
/// hash over the raw HPROF body bytes. Four-byte words are little-endian and
/// unsigned; trailing bytes are passed as signed Java bytes.
fn java_hash_byte_array(bytes: &[u8]) -> i32 {
    let mut h: i32 = 0;
    let len = bytes.len();
    let mut i = 0usize;
    while i + 4 <= len {
        let k = ((bytes[i + 3] as i32) << 24)
            | ((bytes[i + 2] as i32) << 16)
            | ((bytes[i + 1] as i32) << 8)
            | (bytes[i] as i32);
        h = java_mix_h1(h, java_mix_k1(k));
        i += 4;
    }
    while i < len {
        h = java_mix_h1(h, java_mix_k1(bytes[i] as i8 as i32));
        i += 1;
    }
    java_fmix(h, len as i32)
}


/// Sum the aligned bodies of a CSLM's Index/Node objects reachable from
/// `head`. Java's ConcurrentSkipListMapIterator adds these objects' bodies in
/// addition to the container itself.
fn cslm_structure_body(
    head: u64,
    index_map: &HashMap<u64, (u64, u64, u64)>,
    node_map: &HashMap<u64, (u64, u64)>,
    class_by_name: &HashMap<std::sync::Arc<str>, u32>,
    class_aligned: &[i64],
) -> i64 {
    let index_cidx = class_by_name.get("java/util/concurrent/ConcurrentSkipListMap$Index").copied();
    let node_cidx = class_by_name.get("java/util/concurrent/ConcurrentSkipListMap$Node").copied();
    let mut body = 0i64;
    let mut seen_i: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut seen_n: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut istack = vec![head];
    let mut nstack: Vec<u64> = Vec::new();
    while let Some(id) = istack.pop() {
        if id == 0 || !seen_i.insert(id) {
            continue;
        }
        if let Some(idx) = index_cidx {
            body += class_aligned.get(idx as usize).copied().unwrap_or(0);
        }
        if let Some(&(node, down, right)) = index_map.get(&id) {
            if node != 0 { nstack.push(node); }
            if down != 0 { istack.push(down); }
            if right != 0 { istack.push(right); }
        }
    }
    while let Some(id) = nstack.pop() {
        if id == 0 || !seen_n.insert(id) {
            continue;
        }
        if let Some(idx) = node_cidx {
            body += class_aligned.get(idx as usize).copied().unwrap_or(0);
        }
        if let Some(&(next, _)) = node_map.get(&id) {
            if next != 0 { nstack.push(next); }
        }
    }
    body
}

/// Java `VirtualStackTraverser.hashCharArray`.
fn java_hash_char_array(chars: &[u16]) -> i32 {
    let mut h: i32 = 0;
    let mut i = 1usize;
    while i < chars.len() {
        let k = (chars[i - 1] as i32) | ((chars[i] as i32) << 16);
        h = java_mix_h1(h, java_mix_k1(k));
        i += 2;
    }
    if chars.len() % 2 == 1 {
        h ^= java_mix_k1(chars[chars.len() - 1] as i32);
    }
    java_fmix(h, (chars.len() as i32) << 1)
}

fn java_mix_k1(k: i32) -> i32 {
    k.wrapping_mul(-862048943).rotate_left(15).wrapping_mul(461845907)
}

fn java_mix_h1(h: i32, k: i32) -> i32 {
    (h ^ k).rotate_left(13).wrapping_mul(5).wrapping_sub(430675100)
}

fn java_fmix(h: i32, len: i32) -> i32 {
    let mut h = h ^ len;
    h ^= ((h as u32) >> 16) as i32;
    h = h.wrapping_mul(-2048144789);
    h ^= ((h as u32) >> 13) as i32;
    h = h.wrapping_mul(-1028477387);
    h ^ ((h as u32) >> 16) as i32
}

// ---------------------------------------------------------------------------
// Raw primitive-array body walk
// ---------------------------------------------------------------------------
//
// jvm-hprof decodes primitive arrays element by element and exposes neither
// file offsets nor the raw body, so borrowing a body out of the mapped dump
// needs its own walk over the raw stream. The record layout below mirrors
// ss_hprof_parser::record_reader::walk_objects
// (crates/hprof-parser/src/record_reader.rs), the canonical walker; it lives
// here only because this crate does not depend on ss-hprof-parser. Replace it
// with that call once the dependency is declared.

const HPROF_MAGIC: &[u8] = b"JAVA PROFILE 1.0.";
const TAG_HEAP_DUMP: u8 = 0x0C;
const TAG_HEAP_DUMP_SEGMENT: u8 = 0x1C;

/// Bounds-checked reader over a byte region; ids decode at id_size width.
struct RawWalk<'a> {
    data: &'a [u8],
    pos: usize,
    id_size: u8,
}

impl<'a> RawWalk<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| anyhow::anyhow!("hprof walk: offset overflow at {}", self.pos))?;
        if end > self.data.len() {
            anyhow::bail!(
                "hprof walk: truncated at offset {}: need {} bytes, {} left",
                self.pos,
                n,
                self.data.len() - self.pos
            );
        }
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn id(&mut self) -> Result<u64> {
        if self.id_size == 4 {
            return Ok(self.u32()? as u64);
        }
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_be_bytes(a))
    }

    /// Width in bytes of an HPROF basic type code.
    fn field_width(&self, ty: u8) -> Result<usize> {
        Ok(match ty {
            2 => self.id_size as usize,
            4 | 8 => 1,
            5 | 9 => 2,
            6 | 10 => 4,
            7 | 11 => 8,
            other => anyhow::bail!(
                "hprof walk: unknown field type {other:#04x} at offset {}",
                self.pos
            ),
        })
    }

    /// Class dump: class id, trace serial, super id, loader/signers/protection
    /// domain, two reserved ids, instance size, then the constant pool and the
    /// static and instance field descriptors.
    fn skip_class_dump(&mut self) -> Result<()> {
        self.id()?;
        self.u32()?;
        self.id()?;
        self.take(5 * self.id_size as usize)?;
        self.u32()?;
        for _ in 0..self.u16()? {
            self.u16()?;
            let ty = self.u8()?;
            let w = self.field_width(ty)?;
            self.take(w)?;
        }
        for _ in 0..self.u16()? {
            self.id()?;
            let ty = self.u8()?;
            let w = self.field_width(ty)?;
            self.take(w)?;
        }
        for _ in 0..self.u16()? {
            self.id()?;
            self.u8()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bounded scan passes
// ---------------------------------------------------------------------------
//
// Nothing per object is materialised: every pass walks the dump (or a compact
// index built from one walk) and keeps only aggregates, the top-N detail lists
// the report renders and fixed-size buffers. Object bodies are re-read from the
// read-only mapping by offset, and the pages behind the walk cursor are handed
// back to the kernel as the pass advances, so file residency stays inside a
// window instead of reaching the dump's size.

/// Pages kept resident behind the walk cursor.
const SCAN_RELEASE_WINDOW: usize = 24 << 20;

/// Body reads between two release windows: residency then stays near that many
/// pages instead of growing with the dump.
/// How often the on-demand body readers hand the mapping's pages back.
///
/// The duplicate-detection passes read bodies at pseudo-random addresses, so every
/// page they touch stays resident until something releases it. Releasing only every
/// 4096 reads let those pages pile up to a third of the dump; the passes are I/O
/// bound anyway, so a much tighter stride costs little and bounds RSS.
const BODY_RELEASE_STRIDE: usize = 128;

/// One heap-dump object, located in the mapped dump by walk_heap.
enum RawObj {
    /// Instance field bytes at body_off..body_off + body_len.
    Instance { obj_id: u64, class_ref: u64, body_off: usize, body_len: usize },
    /// len identifiers of id_size bytes each, starting at body_off.
    ObjArray { obj_id: u64, class_ref: u64, len: u32, body_off: usize, body_len: usize },
    PrimArray { obj_id: u64, type_code: u8, len: u32, body_off: usize, body_len: usize },
}

/// Hand the mapping's pages below upto back to the kernel.
fn release_before(map: &memmap2::Mmap, upto: usize) {
    if upto == 0 || map.is_empty() {
        return;
    }
    let end = upto.min(map.len());
    // SAFETY: the range is inside a read-only file mapping; a released page is
    // re-read from the file on its next touch, so the bytes never change.
    let _ = unsafe { map.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, 0, end) };
}

/// Walk every heap-dump object in dump order, reporting where its body lives
/// instead of borrowing it, and releasing the pages behind the cursor.
fn walk_heap<F>(mmap: &memmap2::Mmap, data: &[u8], id_size: u8, mut visit: F) -> Result<()>
where
    F: FnMut(RawObj) -> Result<()>,
{
    if id_size != 4 && id_size != 8 {
        anyhow::bail!("hprof walk: id_size must be 4 or 8, got {id_size}");
    }
    let id_bytes = id_size as usize;
    let start = hprof_records_start(data, id_size)?;
    let mut cur = RawWalk { data, pos: start, id_size };
    let mut released = 0usize;
    while cur.pos < data.len() {
        let tag = cur.u8()?;
        cur.u32()?; // time delta in micros
        let len = cur.u32()? as usize;
        let rec_start = cur.pos;
        let rec = cur.take(len)?;
        if tag != TAG_HEAP_DUMP && tag != TAG_HEAP_DUMP_SEGMENT {
            continue;
        }
        let mut sub = RawWalk { data: rec, pos: 0, id_size };
        while sub.pos < rec.len() {
            // A heap-dump segment can span most of the file, so the window has to
            // advance inside it too.
            let abs = rec_start + sub.pos;
            if abs > released + SCAN_RELEASE_WINDOW {
                released = abs - SCAN_RELEASE_WINDOW;
                release_before(mmap, released);
            }
            match sub.u8()? {
                // GC roots. The payload depends on the root kind.
                0xFF | 0x05 | 0x07 => {
                    sub.id()?;
                }
                0x01 => {
                    sub.id()?;
                    sub.id()?;
                }
                0x04 | 0x06 => {
                    sub.id()?;
                    sub.u32()?;
                }
                0x02 | 0x03 | 0x08 => {
                    sub.id()?;
                    sub.u32()?;
                    sub.u32()?;
                }
                0x20 => sub.skip_class_dump()?,
                0x21 => {
                    let obj_id = sub.id()?;
                    sub.u32()?; // stack trace serial
                    let class_ref = sub.id()?;
                    let body_len = sub.u32()? as usize;
                    let body_off = rec_start + sub.pos;
                    sub.take(body_len)?;
                    visit(RawObj::Instance { obj_id, class_ref, body_off, body_len })?;
                }
                0x22 => {
                    let obj_id = sub.id()?;
                    sub.u32()?;
                    let len = sub.u32()?;
                    let class_ref = sub.id()?;
                    let body_off = rec_start + sub.pos;
                    let body_len = len as usize * id_bytes;
                    sub.take(body_len)?;
                    visit(RawObj::ObjArray { obj_id, class_ref, len, body_off, body_len })?;
                }
                0x23 => {
                    let obj_id = sub.id()?;
                    sub.u32()?;
                    let len = sub.u32()?;
                    let type_code = sub.u8()?;
                    let width = sub.field_width(type_code)?;
                    let body_off = rec_start + sub.pos;
                    let body_len = len as usize * width;
                    sub.take(body_len)?;
                    visit(RawObj::PrimArray { obj_id, type_code, len, body_off, body_len })?;
                }
                other => anyhow::bail!(
                    "hprof walk: unknown heap-dump sub-record tag {other:#04x} at offset {abs}"
                ),
            }
        }
    }
    Ok(())
}

/// One primitive array. The body stays in the mapping; only its location is kept.
#[derive(Clone, Copy)]
struct PrimRec {
    obj_id: u64,
    body_off: u64,
    length: u32,
    prim_type: PrimitiveArrayType,
}

impl PrimRec {
    fn elem_size(&self) -> i32 {
        prim_type_size(self.prim_type) as i32
    }

    fn body_len(&self) -> usize {
        self.length as usize * prim_type_size(self.prim_type)
    }
}

/// One object array. The element list stays in the mapping.
#[derive(Clone, Copy)]
struct ObjArrRec {
    obj_id: u64,
    body_off: u64,
    class_idx: u32,
    length: u32,
}

/// Body lookups over the primitive-array index.
struct PrimBodies<'a> {
    map: &'a memmap2::Mmap,
    data: &'a [u8],
    recs: &'a [PrimRec],
    /// Array indices sorted by object id: the walk yields file order.
    order: &'a [u32],
}

impl<'a> PrimBodies<'a> {
    /// Hand the mapping's pages back every few thousand body reads, so a pass
    /// that re-reads bodies out of order still runs inside a bounded window.
    fn pace(&self, i: usize) {
        if i % BODY_RELEASE_STRIDE == 0 {
            release_mapped_pages(self.map);
        }
    }

    fn find(&self, obj_id: u64) -> Option<&'a PrimRec> {
        let mut lo = 0usize;
        let mut hi = self.order.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let rec = &self.recs[self.order[mid] as usize];
            match rec.obj_id.cmp(&obj_id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(rec),
            }
        }
        None
    }

    fn body(&self, rec: &PrimRec) -> &'a [u8] {
        let off = rec.body_off as usize;
        &self.data[off..off + rec.body_len()]
    }

    fn body_of(&self, obj_id: u64) -> Option<(&'a PrimRec, &'a [u8])> {
        let rec = self.find(obj_id)?;
        Some((rec, self.body(rec)))
    }
}

/// One String's content, described instead of copied: the backing array plus the
/// slice of it the String wraps.
#[derive(Clone, Copy)]
struct StrContent {
    array_id: u64,
    enc: u8,
    /// Index of the first unit: a byte offset for LATIN1, a UTF-16 index otherwise.
    start: u32,
    /// Length in UTF-16 units.
    take: u32,
}

/// char[]: big-endian UTF-16 elements.
const STR_ENC_CHAR_BE: u8 = 0;
/// byte[] with coder 0: one LATIN1 character per byte.
const STR_ENC_BYTE_LATIN1: u8 = 1;
/// byte[] with coder 1: little-endian UTF-16 pairs.
const STR_ENC_BYTE_UTF16: u8 = 2;

impl StrContent {
    /// The i-th UTF-16 unit of the content inside body.
    fn unit(&self, body: &[u8], i: usize) -> u16 {
        match self.enc {
            STR_ENC_CHAR_BE => {
                let o = (self.start as usize + i) * 2;
                u16::from_be_bytes([body[o], body[o + 1]])
            }
            STR_ENC_BYTE_LATIN1 => body[self.start as usize + i] as u16,
            _ => {
                let o = (self.start as usize + i) * 2;
                u16::from_le_bytes([body[o], body[o + 1]])
            }
        }
    }

    fn units(&self, body: &[u8]) -> Vec<u16> {
        (0..self.take as usize).map(|i| self.unit(body, i)).collect()
    }

    /// Java StringDeduplicationStats backing contribution for one String.
    fn backing_size(&self, body: &[u8], header_size: i64) -> i64 {
        // arrayLen is the backing array's own length: elements for char[], bytes
        // for byte[] (both Java 9 codings), exactly as the original computed it.
        let (array_len, element_size) = match self.enc {
            STR_ENC_CHAR_BE => (body.len() as i64 / 2, 2i64),
            _ => (body.len() as i64, 1i64),
        };
        let content_len = self.take as i64;
        if array_len == content_len {
            array_shallow_size(array_len as i32, element_size as i32, header_size)
        } else {
            element_size * content_len
        }
    }
}

fn str_content_eq(a: &StrContent, a_body: &[u8], b: &StrContent, b_body: &[u8]) -> bool {
    if a.take != b.take {
        return false;
    }
    (0..a.take as usize).all(|i| a.unit(a_body, i) == b.unit(b_body, i))
}

/// Java hashCharArray over a content descriptor.
fn java_hash_str_content(c: &StrContent, body: &[u8]) -> i32 {
    let mut h: i32 = 0;
    let mut i = 1usize;
    while i < c.take as usize {
        let k = (c.unit(body, i - 1) as i32) | ((c.unit(body, i) as i32) << 16);
        h = java_mix_h1(h, java_mix_k1(k));
        i += 2;
    }
    if c.take % 2 == 1 {
        h ^= java_mix_k1(c.unit(body, c.take as usize - 1) as i32);
    }
    java_fmix(h, (c.take as i32) << 1)
}

/// Resolve a String instance's field bytes to its content descriptor.
///
/// Same reading order as extract_string_content_meta, but the content is left in
/// the mapping: no Vec<u16> copy per String.
fn string_content_ref(
    fields: &[u8],
    class_info: &JavaClassInfo,
    id_bytes: usize,
    prims: &PrimBodies<'_>,
) -> Option<StrContent> {
    let mut offset = 0usize;
    let mut value_ref: Option<u64> = None;
    let mut str_offset: i32 = 0;
    let mut str_count: i32 = i32::MAX;
    let mut coder: u8 = 0;

    for fd in &class_info.instance_fields {
        match fd.type_code {
            2 => {
                if fd.name.as_ref() == "value" {
                    value_ref = read_id_from(fields, offset, id_bytes);
                }
                offset += id_bytes;
            }
            8 => {
                if fd.name.as_ref() == "coder" {
                    coder = fields.get(offset).copied().unwrap_or(0);
                }
                offset += 1;
            }
            10 => {
                if fd.name.as_ref() == "offset" {
                    str_offset = read_be_i32(fields, offset)?;
                } else if fd.name.as_ref() == "count" {
                    str_count = read_be_i32(fields, offset).unwrap_or(i32::MAX);
                }
                offset += 4;
            }
            _ => {
                offset += field_type_size(fd.type_code, id_bytes);
            }
        }
    }

    let value_id = value_ref?;
    let (rec, body) = prims.body_of(value_id)?;
    let enc = match rec.prim_type {
        // Java 9+ coder=0 -> LATIN1 (byte[]), coder=1 -> UTF16 (byte[]).
        // Java 8 -> char[] backing.
        PrimitiveArrayType::Char => STR_ENC_CHAR_BE,
        PrimitiveArrayType::Byte => {
            if coder == 0 {
                STR_ENC_BYTE_LATIN1
            } else {
                STR_ENC_BYTE_UTF16
            }
        }
        _ => return None,
    };
    let start = str_offset.max(0) as usize;
    let available = if enc == STR_ENC_BYTE_LATIN1 { body.len() } else { body.len() / 2 };
    if start > available {
        return None;
    }
    let take = take_count(str_count, available - start);
    Some(StrContent { array_id: value_id, enc, start: start as u32, take: take as u32 })
}

/// One running duplicate-string group (Java JStringStats), holding its
/// representative String by description rather than by a content copy.
struct StrGroup {
    content: StrContent,
    obj_id: u64,
    count: u32,
    backing_count: u32,
    backing_bytes: i64,
}

/// Java StringContentDeduplicator state: a pending map keyed by the 32-bit
/// content hash, plus a content map whose equality check re-reads the
/// representative's backing array out of the mapping.
struct StrGrouping {
    header_size: i64,
    /// Java pending: hash -> the one String seen for that hash, not yet merged.
    pending: HashMap<i32, u32>,
    /// Java content_map: hash -> chain head of materialised groups.
    buckets: HashMap<i32, i32>,
    next: Vec<i32>,
    groups: Vec<StrGroup>,
    /// Java claimedArrays.
    claimed: std::collections::HashSet<u64>,
}

impl StrGrouping {
    fn new(header_size: i64) -> Self {
        Self {
            header_size,
            pending: HashMap::new(),
            buckets: HashMap::new(),
            next: Vec::new(),
            groups: Vec::new(),
            claimed: std::collections::HashSet::new(),
        }
    }

    fn find_materialised(
        &self,
        hash: i32,
        cur: &StrContent,
        cur_body: &[u8],
        prims: &PrimBodies<'_>,
    ) -> Option<u32> {
        let mut idx = self.buckets.get(&hash).copied().unwrap_or(-1);
        while idx >= 0 {
            let g = &self.groups[idx as usize];
            let body = prims.body_of(g.content.array_id).map(|(_, b)| b).unwrap_or(&[]);
            if str_content_eq(&g.content, body, cur, cur_body) {
                return Some(idx as u32);
            }
            idx = self.next[idx as usize];
        }
        None
    }

    fn open(&mut self, hash: i32, gi: usize) {
        let head = self.buckets.get(&hash).copied().unwrap_or(-1);
        self.next[gi] = head;
        self.buckets.insert(hash, gi as i32);
    }

    /// Java StringContentDeduplicator.processString.
    fn observe(&mut self, hash: i32, cur: StrContent, obj_id: u64, backing: i64, prims: &PrimBodies<'_>) {
        let cur_body = prims.body_of(cur.array_id).map(|(_, b)| b).unwrap_or(&[]);
        // Java materialises a collided pending entry, then looks the current
        // content up in content_map rather than creating a fresh pending entry.
        let mut collided = false;
        if let Some(pi) = self.pending.remove(&hash) {
            let first = self.groups[pi as usize].content;
            let first_body = prims.body_of(first.array_id).map(|(_, b)| b).unwrap_or(&[]);
            let first_backing = first.backing_size(first_body, self.header_size);
            if str_content_eq(&first, first_body, &cur, cur_body) {
                // Second occurrence: Java folds the pair into content_map with
                // count 2 and counts the first backing array unconditionally.
                let claimed = self.claimed.insert(cur.array_id);
                let g = &mut self.groups[pi as usize];
                g.count = 2;
                g.backing_count = 1;
                g.backing_bytes = first_backing;
                if claimed {
                    g.backing_count += 1;
                    g.backing_bytes += backing;
                }
                self.open(hash, pi as usize);
                return;
            }
            // 32-bit hash collision.
            let claimed_cur = self.claimed.insert(cur.array_id);
            {
                let g = &mut self.groups[pi as usize];
                g.backing_count = 1;
                g.backing_bytes = first_backing;
            }
            self.open(hash, pi as usize);
            if let Some(eq) = self.find_materialised(hash, &cur, cur_body, prims) {
                let g = &mut self.groups[eq as usize];
                g.count += 1;
                if claimed_cur {
                    g.backing_count += 1;
                    g.backing_bytes += backing;
                }
                return;
            }
            collided = true;
        } else if let Some(eq) = self.find_materialised(hash, &cur, cur_body, prims) {
            let claimed = self.claimed.insert(cur.array_id);
            let g = &mut self.groups[eq as usize];
            g.count += 1;
            if claimed {
                g.backing_count += 1;
                g.backing_bytes += backing;
            }
            return;
        }

        if collided {
            let claimed = self.claimed.insert(cur.array_id);
            let gi = self.groups.len();
            self.groups.push(StrGroup {
                content: cur,
                obj_id,
                count: 1,
                backing_count: u32::from(claimed),
                backing_bytes: if claimed { backing } else { 0 },
            });
            self.next.push(-1);
            self.open(hash, gi);
        } else {
            // First content for this hash: Java claims the array but keeps the
            // content in pending, so no backing is attributed yet.
            self.claimed.insert(cur.array_id);
            let gi = self.groups.len();
            self.groups.push(StrGroup { content: cur, obj_id, count: 1, backing_count: 0, backing_bytes: 0 });
            self.next.push(-1);
            self.pending.insert(hash, gi as u32);
        }
    }

    /// Java content_map.size() + pending.size(): every group is in one of the two.
    fn unique_count(&self) -> usize {
        self.groups.len()
    }

    /// The report's duplicate groups, i.e. materialised entries with count > 1.
    fn report_groups(
        &self,
        prims: &PrimBodies<'_>,
        string_body_size: i64,
    ) -> Vec<ss_stats::duplicate_strings::DupStringGroup> {
        let mut out = Vec::new();
        for (gi, g) in self.groups.iter().enumerate() {
            prims.pace(gi);
            if g.count <= 1 {
                continue;
            }
            let body = prims.body_of(g.content.array_id).map(|(_, b)| b).unwrap_or(&[]);
            let extra = (g.count - 1) as i64;
            let mut wasted = extra * string_body_size;
            if g.backing_count > 1 {
                wasted += (g.backing_bytes / g.backing_count as i64) * (g.backing_count as i64 - 1);
            }
            out.push(ss_stats::duplicate_strings::DupStringGroup {
                content: String::from_utf16_lossy(&g.content.units(body)),
                count: g.count as usize,
                wasted_bytes: wasted,
                backing_array_count: g.backing_count as usize,
                sample_obj_ids: vec![g.obj_id],
            });
        }
        out
    }
}


/// Body lookups over the object-array index.
struct ObjBodies<'a> {
    map: &'a memmap2::Mmap,
    data: &'a [u8],
    recs: &'a [ObjArrRec],
    /// Array indices sorted by object id: the walk yields file order.
    order: &'a [u32],
    id_bytes: usize,
}

impl<'a> ObjBodies<'a> {
    fn pace(&self, i: usize) {
        if i % BODY_RELEASE_STRIDE == 0 {
            release_mapped_pages(self.map);
        }
    }

    fn find(&self, obj_id: u64) -> Option<&'a ObjArrRec> {
        let mut lo = 0usize;
        let mut hi = self.order.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let rec = &self.recs[self.order[mid] as usize];
            match rec.obj_id.cmp(&obj_id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(rec),
            }
        }
        None
    }

    /// The non-null element addresses of one object array, in dump order.
    fn elems(&self, rec: &ObjArrRec) -> Vec<u64> {
        let off = rec.body_off as usize;
        let mut out = Vec::with_capacity(rec.length as usize);
        for i in 0..rec.length as usize {
            let o = off + i * self.id_bytes;
            let id = if self.id_bytes == 8 {
                u64::from_be_bytes(self.data[o..o + 8].try_into().unwrap())
            } else {
                u32::from_be_bytes(self.data[o..o + 4].try_into().unwrap()) as u64
            };
            if id != 0 {
                out.push(id);
            }
        }
        out
    }
}

/// Offset of the first top-level record, skipping the header when present.
fn hprof_records_start(data: &[u8], id_size: u8) -> Result<usize> {
    if !data.starts_with(HPROF_MAGIC) {
        return Ok(0);
    }
    let mut cur = RawWalk { data, pos: HPROF_MAGIC.len(), id_size };
    let version = cur.u8()?;
    if version != b'2' && version != b'3' {
        anyhow::bail!("hprof walk: unsupported HPROF version 1.0.{}", version as char);
    }
    if cur.u8()? != 0 {
        anyhow::bail!("hprof walk: expected NUL terminator after the version string");
    }
    let header_id_size = cur.u32()?;
    if header_id_size != id_size as u32 {
        anyhow::bail!("hprof walk: id_size {id_size} does not match the header's {header_id_size}");
    }
    cur.take(8)?; // timestamp in millis since epoch
    Ok(cur.pos)
}

/// Visit (obj_id, prim_type_code, element_count, body) for every primitive
/// array dump in data, in file order. The body borrows data directly.
/// Collect every primitive array in file order, borrowing each body from the
/// mapped dump instead of copying it.
/// Shallow size of an array object, matching Java's
/// `PrimitiveArrayInstance.getBodySize()` / `ObjectArrayInstance.getBodySize()`:
///   `align(elementSize * length + objectAlignBase, alignment)`
/// where `objectAlignBase = headerSize + arrayLengthField(4)`. Rust's header_size
/// is the object header (e.g. 12 with compressed oops); arrays additionally
/// carry a 4-byte length field, so the base is `header_size + 4`.
fn array_shallow_size(length: i32, element_size: i32, header_size: i64) -> i64 {
    let object_align_base = header_size + 4;
    let data_size = length as i64 * element_size as i64;
    let raw = object_align_base + data_size;
    // Align to 8 bytes (Java's default object alignment).
    (raw + 7) & !7
}

fn boxed_primitive_size(name: &str) -> i32 {
    match name {
        n if n == "java/lang/Byte" || n == "java.lang.Byte"
            || n == "java/lang/Boolean" || n == "java.lang.Boolean" => 1,
        n if n == "java/lang/Short" || n == "java.lang.Short"
            || n == "java/lang/Character" || n == "java.lang.Character" => 2,
        n if n == "java/lang/Integer" || n == "java.lang.Integer"
            || n == "java/lang/Float" || n == "java.lang.Float" => 4,
        n if n == "java/lang/Long" || n == "java.lang.Long"
            || n == "java/lang/Double" || n == "java.lang.Double" => 8,
        _ => 0,
    }
}

#[derive(Clone, Copy)]
struct PrimDupNode {
    /// Representative array: the one that created this node.
    array_idx: u32,
    count: u32,
    next: i32,
}

/// One PrimitiveArraySizeAccumulator: pending first-seen hashes plus the
/// hash -> node chain map, exactly as in the Java reference. The per-array group
/// ids the Java code threads through are not kept: nothing observable reads them.
struct PrimDupAccumulator {
    group_map: HashMap<i32, u32>,
    pending: HashMap<i32, u32>,
    nodes: Vec<PrimDupNode>,
    /// Members of hash-0 nodes, kept for the opt-in forensic dump only.
    hash0_members: HashMap<u32, Vec<u64>>,
    per_type_total: [u32; 8],
    per_type_distinct: [u32; 8],
    per_type_waste: [i64; 8],
}

impl PrimDupAccumulator {
    fn new() -> Self {
        Self {
            group_map: HashMap::new(),
            pending: HashMap::new(),
            nodes: Vec::new(),
            hash0_members: HashMap::new(),
            per_type_total: [0; 8],
            per_type_distinct: [0; 8],
            per_type_waste: [0; 8],
        }
    }

    fn add_node(&mut self, array_idx: u32, count: u32, hash: i32, members: &[u64]) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(PrimDupNode { array_idx, count, next: -1 });
        if hash == 0 {
            self.hash0_members.insert(idx, members.to_vec());
        }
        idx
    }

    fn add_member(&mut self, node_idx: u32, obj_id: u64) {
        if let Some(members) = self.hash0_members.get_mut(&node_idx) {
            members.push(obj_id);
        }
    }
}

/// Mirror of PrimitiveArraySizeAccumulator.processArray. Bodies are compared out
/// of the mapping, so nothing per array is held beyond the accumulator's chains.
fn process_prim_dup_array(
    acc: &mut PrimDupAccumulator,
    prims: &PrimBodies<'_>,
    prim_hashes: &[i32],
    prim_class_id: &[u64; 8],
    idx: usize,
) {
    let arr = prims.recs[idx];
    let hash = prim_hashes[idx];
    let type_idx = prim_type_code_index(arr.prim_type);
    let arr_class_id = prim_class_id[type_idx];
    let arr_body = prims.body(&arr);
    acc.per_type_total[type_idx] += 1;

    if let Some(first_idx) = acc.pending.remove(&hash) {
        let first = prims.recs[first_idx as usize];
        let first_class_id = prim_class_id[prim_type_code_index(first.prim_type)];
        let equal = first_class_id == arr_class_id
            && first.length == arr.length
            && prims.body(&first) == arr_body;
        if equal {
            let node = acc.add_node(idx as u32, 2, hash, &[first.obj_id, arr.obj_id]);
            acc.group_map.insert(hash, node);
        } else {
            let first_node = acc.add_node(first_idx, 1, hash, &[first.obj_id]);
            let second_node = acc.add_node(idx as u32, 1, hash, &[arr.obj_id]);
            acc.nodes[first_node as usize].next = second_node as i32;
            acc.per_type_distinct[type_idx] += 1;
            acc.group_map.insert(hash, first_node);
        }
        return;
    }

    if let Some(&head) = acc.group_map.get(&hash) {
        let mut cur = head as i32;
        let mut last: i32 = -1;
        loop {
            let node = acc.nodes[cur as usize];
            let rep = prims.recs[node.array_idx as usize];
            let rep_class_id = prim_class_id[prim_type_code_index(rep.prim_type)];
            if rep_class_id == arr_class_id && rep.length == arr.length && prims.body(&rep) == arr_body {
                acc.nodes[cur as usize].count += 1;
                acc.add_member(cur as u32, arr.obj_id);
                return;
            }
            last = cur;
            if node.next < 0 {
                break;
            }
            cur = node.next;
        }
        let node = acc.add_node(idx as u32, 1, hash, &[arr.obj_id]);
        acc.nodes[last as usize].next = node as i32;
        acc.per_type_distinct[type_idx] += 1;
        return;
    }

    // First array with this hash: remember it, no group node yet.
    acc.pending.insert(hash, idx as u32);
    acc.per_type_distinct[type_idx] += 1;
}

/// Java's per-primitive-type display names, indexed by Java's primitive type
/// code order (boolean, byte, char, short, int, float, long, double).
const PRIM_TYPE_DISPLAY: [&str; 8] = [
    "boolean[]",
    "byte[]",
    "char[]",
    "short[]",
    "int[]",
    "float[]",
    "long[]",
    "double[]",
];

fn detect_duplicate_prim_arrays(
    prims: &PrimBodies<'_>,
    prim_hashes: &[i32],
    prim_class_id: &[u64; 8],
    header_size: i64,
    string_backing: &std::collections::HashSet<u64>,
    container_backing: &std::collections::HashSet<u64>,
    stats: &StatsEngine,
) {
    let debug_on = ss_stats::debug_dump::enabled();
    let mut pass2 = PrimDupAccumulator::new();
    let mut pass1 = PrimDupAccumulator::new();

    // Pass 2: every non-char primitive array, in dump order.
    for (idx, rec) in prims.recs.iter().enumerate() {
        prims.pace(idx);
        if rec.prim_type == PrimitiveArrayType::Char {
            continue;
        }
        process_prim_dup_array(&mut pass2, prims, prim_hashes, prim_class_id, idx);
    }

    // Pass 1: char arrays that String/StringBuilder processing did not claim.
    for (idx, rec) in prims.recs.iter().enumerate() {
        prims.pace(idx);
        if rec.prim_type != PrimitiveArrayType::Char || string_backing.contains(&rec.obj_id) {
            continue;
        }
        process_prim_dup_array(&mut pass1, prims, prim_hashes, prim_class_id, idx);
    }

    // Java's mergeAndBuildReport() iterates only the first bucket of each
    // groupMap. fastutil stores key 0 in slot 0, so the first bucket is the
    // hash-0 chain (all empty arrays hash to 0).
    let mut per_class_entries: Vec<ss_stats::duplicate_arrays::DupArrayClassEntry> = Vec::new();
    let mut top_groups: Vec<ss_stats::duplicate_arrays::DupArrayGroup> = Vec::new();
    let mut debug_groups: Vec<ss_stats::debug_dump::DupPrimGroupDebug> = Vec::new();
    let mut iterated_nodes: usize = 0;
    let mut arrays_in_groups: usize = 0;
    let mut waste_total: i64 = 0;

    for acc in [&pass2, &pass1] {
        let mut type_waste = acc.per_type_waste;
        if let Some(&head) = acc.group_map.get(&0) {
            let mut cur = head as i32;
            while cur >= 0 {
                iterated_nodes += 1;
                let node = acc.nodes[cur as usize];
                if node.count > 1 {
                    let rep = prims.recs[node.array_idx as usize];
                    let body = array_shallow_size(rep.length as i32, rep.elem_size(), header_size);
                    let wasted = body * (node.count as i64 - 1);
                    waste_total += wasted;
                    arrays_in_groups += node.count as usize;
                    let type_idx = prim_type_code_index(rep.prim_type);
                    type_waste[type_idx] += wasted;

                    let class_name = prim_type_display_name(rep.prim_type).to_string();
                    if debug_on {
                        let members = acc.hash0_members.get(&(cur as u32));
                        debug_groups.push(ss_stats::debug_dump::DupPrimGroupDebug {
                            class_name: class_name.clone(),
                            array_length: rep.length as i32,
                            element_size: rep.elem_size(),
                            body_size: body,
                            count: node.count as usize,
                            wasted_bytes: wasted,
                            content_hash: ss_stats::debug_dump::fnv1a64(prims.body(&rep)),
                            members: members
                                .map(|ids| {
                                    ids.iter()
                                        .map(|obj_id| ss_stats::debug_dump::DupPrimMemberDebug {
                                            obj_id: *obj_id,
                                            string_backing: string_backing.contains(obj_id),
                                            container_backing: container_backing.contains(obj_id),
                                        })
                                        .collect()
                                })
                                .unwrap_or_default(),
                        });
                    }

                    top_groups.push(ss_stats::duplicate_arrays::DupArrayGroup {
                        class_name,
                        count: node.count as usize,
                        wasted_bytes: wasted,
                        size_bytes: body,
                        description: format!(
                            "{}[{}]",
                            PRIM_TYPE_DISPLAY[type_idx].trim_end_matches("[]"),
                            rep.length
                        ),
                        sample_obj_ids: vec![rep.obj_id],
                    });
                }
                cur = node.next;
            }
        }

        // Java's DuplicateClassEntry.instanceCount is the class' total instance
        // count, available to both accumulators; only types with
        // duplicateSize > 0 are reported.
        for type_idx in 0..8 {
            if type_waste[type_idx] > 0 {
                per_class_entries.push(ss_stats::duplicate_arrays::DupArrayClassEntry {
                    class_name: PRIM_TYPE_DISPLAY[type_idx].to_string(),
                    instance_count: acc.per_type_total[type_idx] as usize,
                    unique_count: acc.per_type_distinct[type_idx] as usize,
                    wasted_bytes: type_waste[type_idx],
                });
            }
        }
    }

    // Java's totalArrayCount only counts arrays that entered an accumulator:
    // all non-char arrays + char arrays not claimed by String/StringBuilder.
    let total_processed: usize = pass2.per_type_total.iter().map(|&v| v as usize).sum::<usize>()
        + pass1.per_type_total.iter().map(|&v| v as usize).sum::<usize>();

    // report.totalGroupCount = enumerated nodes + both pending maps.
    let unique_count = iterated_nodes + pass2.pending.len() + pass1.pending.len();

    eprintln!(
        "Duplicate prim arrays: {} processed, {} unique/group count, {} reported groups, {} bytes waste",
        total_processed,
        unique_count,
        top_groups.len(),
        waste_total
    );

    if debug_on {
        ss_stats::debug_dump::record_dup_prim_totals(ss_stats::debug_dump::DupPrimScanTotals {
            scanned_arrays: total_processed,
            eligible_arrays: total_processed,
            arrays_in_groups,
            groups: debug_groups.len(),
            waste_bytes: waste_total,
            unique_count,
        });
        ss_stats::debug_dump::record_dup_prim_groups(debug_groups);
    }

    stats
        .dup_prim_arrays
        .set_results(total_processed, unique_count, per_class_entries, top_groups);
}

fn detect_duplicate_obj_arrays(
    objs: &ObjBodies<'_>,
    classes: &[JavaClassInfo],
    stats: &StatsEngine,
    header_size: i64,
    oop_size: i32,
) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Group by (class_idx, refs_hash), only for arrays with content. Keys are
    // chained through a flat next array instead of one Vec per key: a key with a
    // single array can never produce a group.
    let mut heads: HashMap<(u32, u64), u32> = HashMap::new();
    let mut next: Vec<i32> = Vec::new();
    let mut members: Vec<u32> = Vec::new();
    for (i, rec) in objs.recs.iter().enumerate() {
        objs.pace(i);
        if rec.length <= 4 {
            continue;
        }
        let refs = objs.elems(rec);
        let mut hasher = DefaultHasher::new();
        refs.hash(&mut hasher);
        let hash = hasher.finish();
        let key = (rec.class_idx, hash);
        let mi = members.len() as u32;
        members.push(i as u32);
        next.push(-1);
        match heads.get_mut(&key) {
            Some(head) => {
                next[mi as usize] = *head as i32;
                *head = mi;
            }
            None => {
                heads.insert(key, mi);
            }
        }
    }

    let mut per_class: HashMap<u32, (usize, usize, i64)> = HashMap::new();
    let mut top_groups: Vec<ss_stats::duplicate_arrays::DupArrayGroup> = Vec::new();
    let mut total_count: usize = 0;

    let mut paced = 0usize;
    for ((class_idx, _), head) in &heads {
        // Members of one key, back in dump order.
        let mut chain: Vec<u32> = Vec::new();
        let mut cur = *head as i32;
        while cur >= 0 {
            paced += 1;
            objs.pace(paced);
            chain.push(members[cur as usize]);
            cur = next[cur as usize];
        }
        chain.reverse();
        if chain.len() < 2 {
            continue;
        }

        let mut confirmed_groups: Vec<Vec<u32>> = Vec::new();
        for &ai in &chain {
            paced += 1;
            objs.pace(paced);
            let refs = objs.elems(&objs.recs[ai as usize]);
            let mut placed = false;
            for group in confirmed_groups.iter_mut() {
                if objs.elems(&objs.recs[group[0] as usize]) == refs {
                    group.push(ai);
                    placed = true;
                    break;
                }
            }
            if !placed {
                confirmed_groups.push(vec![ai]);
            }
        }

        for group in confirmed_groups {
            if group.len() < 2 {
                continue;
            }
            let count = group.len();
            total_count += count;
            let rep = objs.recs[group[0] as usize];
            let size = array_shallow_size(rep.length as i32, oop_size, header_size);
            let wasted = size * (count as i64 - 1);

            let class_name = if (*class_idx as usize) < classes.len() {
                classes[*class_idx as usize].name.replace('/', ".")
            } else {
                "unknown".to_string()
            };

            let (total, unique, waste) = per_class.entry(*class_idx).or_insert((0, 0, 0));
            *total += count;
            *unique += 1;
            *waste += wasted;

            if wasted > 1024 {
                let samples: Vec<u64> = group.iter().take(5).map(|&ai| objs.recs[ai as usize].obj_id).collect();
                top_groups.push(ss_stats::duplicate_arrays::DupArrayGroup {
                    class_name: class_name.clone(),
                    count,
                    wasted_bytes: wasted,
                    size_bytes: size,
                    description: format!("obj[{}]", rep.length),
                    sample_obj_ids: samples,
                });
            }
        }
    }

    let per_class_entries: Vec<ss_stats::duplicate_arrays::DupArrayClassEntry> = per_class
        .iter()
        .filter(|(_, (total, _, _))| *total > 1)
        .map(|(class_idx, (total, unique, waste))| {
            let name = if (*class_idx as usize) < classes.len() {
                classes[*class_idx as usize].name.replace('/', ".")
            } else {
                "unknown".to_string()
            };
            ss_stats::duplicate_arrays::DupArrayClassEntry {
                class_name: name,
                instance_count: *total,
                unique_count: *unique,
                wasted_bytes: *waste,
            }
        })
        .collect();

    let unique_total: usize = per_class_entries.iter().map(|e| e.unique_count).sum();
    eprintln!(
        "Duplicate obj arrays: {} total in {} groups",
        total_count,
        per_class_entries.len()
    );

    stats.dup_obj_arrays.set_results(total_count, unique_total, per_class_entries, top_groups);
}

fn extract_capacity_field(
    fields: &[u8],
    class_info: &JavaClassInfo,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> i32 {
    let mut offset = 0usize;
    let mut current_id = class_info.class_object_id;

    loop {
        let cls = if current_id == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&current_id).and_then(|&idx| all_classes.get(idx as usize))
        };

        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.name.as_ref() == "capacity" && fd.type_code == 10 {
                    return read_be_i32(fields, offset).unwrap_or(0);
                }
                offset += field_type_size(fd.type_code, 8);
            }
            if cls.super_class_id != 0 {
                current_id = cls.super_class_id;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    0
}

/// Read the `address` field of a DirectByteBuffer (a long), walking the class
/// hierarchy. Returns 0 if absent. Mirrors Java's `addressFieldIndex` lookup.
fn extract_address_field(
    fields: &[u8],
    class_info: &JavaClassInfo,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> u64 {
    let mut offset = 0usize;
    let mut current_id = class_info.class_object_id;
    let id_size = current_id_size();
    let mut base_count: Option<i64> = None;

    loop {
        let cls = if current_id == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&current_id).and_then(|&idx| all_classes.get(idx as usize))
        };
        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                // `address` is stored as an 8-byte long. On JDK 17 it shows up
                // as HPROF type_code 11 (8-byte), on older JDKs as 7. Accept
                // either; both are 8 bytes wide per field_type_size.
                if fd.name.as_ref() == "address" && (fd.type_code == 11 || fd.type_code == 7) {
                    return read_be_u64_id(fields, offset, 8).unwrap_or(0);
                }
                offset += field_type_size(fd.type_code, id_size);
            }
            if cls.super_class_id != 0 {
                current_id = cls.super_class_id;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    0
}

/// Whether the instance declares a field named `name`. Used to skip
/// DirectByteBuffers carrying `fd`/`att` (Java analyzeBuffer lines 73-78).
fn has_field(
    class_info: &JavaClassInfo,
    name: &str,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> bool {
    let mut current_id = class_info.class_object_id;
    loop {
        let cls = if current_id == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&current_id).and_then(|&idx| all_classes.get(idx as usize))
        };
        if let Some(cls) = cls {
            if cls.instance_fields.iter().any(|fd| fd.name.as_ref() == name) {
                return true;
            }
            if cls.super_class_id != 0 {
                current_id = cls.super_class_id;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    false
}

fn current_id_size() -> usize {
    ID_SIZE.with(|c| c.get())
}

thread_local! {
    static ID_SIZE: std::cell::Cell<usize> = const { std::cell::Cell::new(8) };
}

/// Whether the object-reference field named `name` is non-null in this
/// instance. Used to skip DirectByteBuffers whose `fd`/`att` is set (Java
/// analyzeBuffer lines 73-78). Walks the class hierarchy to locate the field,
/// reads the raw id bytes, and returns true only if they are non-zero.
fn ref_field_nonnull(
    fields: &[u8],
    class_info: &JavaClassInfo,
    name: &str,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> bool {
    let mut offset = 0usize;
    let mut current_id = class_info.class_object_id;
    let id_size = current_id_size();
    let mut base_count: Option<i64> = None;

    loop {
        let cls = if current_id == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&current_id).and_then(|&idx| all_classes.get(idx as usize))
        };
        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.name.as_ref() == name && fd.type_code == 2 {
                    // Object reference — read id_size bytes big-endian.
                    if offset + id_size <= fields.len() {
                        let raw = read_be_u64_id(fields, offset, id_size).unwrap_or(0);
                        return raw != 0;
                    }
                    return false;
                }
                offset += field_type_size(fd.type_code, id_size);
            }
            if cls.super_class_id != 0 {
                current_id = cls.super_class_id;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    false
}

fn read_be_u64_id(fields: &[u8], offset: usize, id_size: usize) -> Option<u64> {
    if id_size == 8 && offset + 8 <= fields.len() {
        Some(u64::from_be_bytes([
            fields[offset], fields[offset+1], fields[offset+2], fields[offset+3],
            fields[offset+4], fields[offset+5], fields[offset+6], fields[offset+7],
        ]))
    } else if offset + 4 <= fields.len() {
        Some(u32::from_be_bytes([fields[offset], fields[offset+1], fields[offset+2], fields[offset+3]]) as u64)
    } else {
        None
    }
}

fn aggregate_direct_buffers(
    instances: &[DirectBufferData],
    classes: &[JavaClassInfo],
    stats: &StatsEngine,
) {
    use std::collections::{HashMap as StdMap, HashSet};
    let mut per_class: StdMap<u32, (usize, i64)> = StdMap::new(); // class_idx -> (count, total_capacity)
    let mut total_count: usize = 0;
    let mut total_capacity: i64 = 0;
    // Dedup by native address: a DirectByteBufferR view points at the same
    // native block as its underlying DirectByteBuffer. Only the first buffer
    // for each address contributes capacity (mirrors Java's seenAddresses).
    let mut seen_addresses: HashSet<u64> = HashSet::new();
    // Per-buffer deduped records for reachable/unreachable splitting later.
    let mut buffer_records: Vec<ss_stats::direct_buffer::DirectBufferRecord> = Vec::new();

    for inst in instances {
        total_count += 1;
        // If this buffer shares an address with one already counted, skip it.
        let mut cap = inst.capacity as i64;
        if inst.address != 0 && !seen_addresses.insert(inst.address) {
            cap = 0;
        }
        total_capacity += cap;
        let (count, total) = per_class.entry(inst.class_idx).or_insert((0, 0));
        *count += 1;
        *total += cap;
        let class_name = if (inst.class_idx as usize) < classes.len() {
            classes[inst.class_idx as usize].name.replace('/', ".")
        } else {
            "unknown".to_string()
        };
        buffer_records.push(ss_stats::direct_buffer::DirectBufferRecord {
            obj_id: inst.obj_id,
            capacity: cap,
            class_name,
        });
    }

    if total_count == 0 { return; }

    let per_class_entries: Vec<ss_stats::direct_buffer::DirectBufferClassEntry> = per_class
        .iter()
        .map(|(class_idx, (count, cap))| {
            let name = if (*class_idx as usize) < classes.len() {
                classes[*class_idx as usize].name.replace('/', ".")
            } else {
                "unknown".to_string()
            };
            ss_stats::direct_buffer::DirectBufferClassEntry {
                class_name: name,
                count: *count,
                total_capacity: *cap,
            }
        })
        .collect();

    // Top N by capacity for per-instance ref chain rendering.
    let mut sorted: Vec<&DirectBufferData> = instances.iter().collect();
    sorted.sort_by(|a, b| b.capacity.cmp(&a.capacity));
    let sample_obj_ids: Vec<u64> = sorted.iter().take(20).map(|d| d.obj_id).collect();

    eprintln!(
        "DirectByteBuffers: {} instances, {} off-heap capacity, {} distinct addresses",
        total_count,
        total_capacity / (1024 * 1024),
        seen_addresses.len(),
    );

    stats.direct_buffers.set_results(
        total_count,
        total_capacity,
        per_class_entries,
        sample_obj_ids,
        buffer_records,
    );
}

/// Push an obj_id to a sample list, capping at 50 entries. Used to capture
/// samples of bad arrays for per-instance reference chain rendering.
fn push_sample(samples: &Mutex<Vec<u64>>, obj_id: u64) {
    let mut v = samples.lock().unwrap();
    if v.len() < 50 {
        v.push(obj_id);
    }
}

/// Java's PrimitiveArrayInstance.computeZeroStats():
/// (trailing zero elements, elements where every byte is zero,
///  elements whose high half is all zero).
fn compute_zero_stats(raw_body: &[u8], element_size: usize) -> (usize, usize, usize) {
    if element_size == 0 {
        return (0, 0, 0);
    }
    let mut trailing_bytes = 0usize;
    for &b in raw_body.iter().rev() {
        if b == 0 {
            trailing_bytes += 1;
        } else {
            break;
        }
    }
    let trailing = trailing_bytes / element_size;

    let mut zero_elements = 0usize;
    let mut high_bits_zero = 0usize;
    for chunk in raw_body.chunks_exact(element_size) {
        if chunk.iter().all(|&b| b == 0) {
            zero_elements += 1;
        }
        if element_size > 1 {
            let half = element_size / 2;
            if chunk[..half].iter().all(|&b| b == 0) {
                high_bits_zero += 1;
            }
        }
    }
    (trailing, zero_elements, high_bits_zero)
}

/// Java's `JavaClassInfo.isPrimitiveArrayType()`: name is one or more '['
/// followed by a single primitive type code. `[I` and `[[I` both match; the
/// latter is also `arrayDimensions > 1`, so `isObjectArray()` is false.
fn is_primitive_array_class_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes[0] != b'[' {
        return false;
    }
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] == b'[' {
        i += 1;
    }
    i + 1 == bytes.len()
        && matches!(bytes[i], b'Z' | b'B' | b'C' | b'S' | b'I' | b'J' | b'F' | b'D')
}

/// Classify a primitive array using Java's processPrimitiveArray order and
/// waste formulas. Returns (kind, waste) for the first matching category.
fn classify_prim_array(
    length: i32,
    raw_body: &[u8],
    element_size: i32,
    config: &ScanConfig,
    full_size: i64,
) -> Option<(u8, i64)> {
    use ss_stats::problem_types::*;
    if length < 0 || element_size <= 0 {
        return None;
    }
    if length == 0 {
        return Some((PRIM_ARRAY_LEN_ZERO, full_size));
    }
    if length == 1 {
        return Some((PRIM_ARRAY_LEN_ONE, full_size));
    }

    let len = length as usize;
    let esz = element_size as i64;
    let (trailing, zero_elements, high_bits_zero) = compute_zero_stats(raw_body, element_size as usize);

    if trailing >= len {
        return Some((PRIM_ARRAY_ALL_ZERO, full_size));
    }
    if (trailing as f64) > (config.min_trailing_zeroes_percent / 100.0) * len as f64 {
        return Some((PRIM_ARRAY_TRAIL_ZEROES, trailing as i64 * esz));
    }
    if (zero_elements as f64) > (config.min_sparse_zeroes_percent / 100.0) * len as f64 {
        return Some((PRIM_ARRAY_SPARSE, zero_elements as i64 * esz));
    }
    if element_size > 1
        && (high_bits_zero as f64) > (config.min_unused_hi_bits_percent / 100.0) * len as f64
    {
        return Some((PRIM_ARRAY_ZERO_HIGH_BITS, high_bits_zero as i64 * esz / 2));
    }
    None
}

impl LinearScanCoordinator {
    /// Extract thread names by walking Thread instances in the HPROF.
    /// Returns Vec<ThreadInfo> with name, daemon, priority for each thread.
    ///
    /// Each thread object and each String it names is read by address through the
    /// object directory; the dump is neither mapped nor scanned.
    pub fn extract_thread_info(&self, hprof_path: &Path) -> Vec<ss_heap_model::heap_dump::ThreadInfo> {
        let bodies = match AddrBodies::open(&self.heap, hprof_path) {
            Some(b) => b,
            None => return Vec::new(),
        };
        let id_bytes = bodies.id_bytes;

        // Build class_object_id -> class_idx
        let mut class_obj_to_idx: HashMap<u64, u32> = HashMap::new();
        for cls in &self.heap.classes {
            if cls.class_object_id != 0 {
                class_obj_to_idx.insert(cls.class_object_id, cls.index);
            }
        }

        // Collect the set of thread object addresses we need
        let thread_obj_addrs: HashMap<u64, u32> = self.heap.thread_serial_map.iter()
            .map(|(&serial, &addr)| (addr, serial))
            .collect();

        if thread_obj_addrs.is_empty() {
            return Vec::new();
        }

        // Thread.name -> String -> char[]/byte[] -> content, plus the
        // Thread.daemon (boolean) and Thread.priority (int) fields.
        let thread_classes: &[&str] = &[
            "java/lang/Thread",
            "java/lang/VirtualThread",
        ];
        let mut thread_class_indices: Vec<u32> = Vec::new();
        for &tc in thread_classes {
            if let Some(&idx) = self.heap.class_by_name.get(tc) {
                thread_class_indices.push(idx);
            }
        }

        if thread_class_indices.is_empty() {
            eprintln!("Thread info: no Thread class found");
            return Vec::new();
        }

        // Resolve thread names one object graph at a time.
        let mut thread_info_vec: Vec<ss_heap_model::heap_dump::ThreadInfo> = Vec::new();

        for (&addr, &serial) in &thread_obj_addrs {
            let mut name = format!("<not found @ 0x{:x}>", addr);
            let mut daemon = false;
            let mut priority: i32 = 5;

            if let Some(body) = bodies.get(addr) {
                if (body.class_idx as usize) < self.heap.classes.len() {
                    let class_info = &self.heap.classes[body.class_idx as usize];

                    // Walk class hierarchy to find the name field
                    let name_ref = find_field_ref(&body.bytes, class_info, "name", id_bytes, &class_obj_to_idx, &self.heap.classes);

                    // Also extract daemon and priority
                    daemon = read_boolean_field(&body.bytes, class_info, "daemon", id_bytes, &class_obj_to_idx, &self.heap.classes);
                    priority = read_int_field(&body.bytes, class_info, "priority", id_bytes, &class_obj_to_idx, &self.heap.classes);

                    if let Some(name_ref) = name_ref {
                        name = resolve_string_body(&bodies, name_ref, id_bytes, &class_obj_to_idx, &self.heap.classes)
                            .unwrap_or_else(|| format!("<unnamed @ 0x{:x}>", addr));
                    } else {
                        name = format!("<unnamed @ 0x{:x}>", addr);
                    }
                }
            }

            thread_info_vec.push(ss_heap_model::heap_dump::ThreadInfo {
                thread_serial: serial,
                thread_obj_addr: addr,
                name,
                is_daemon: daemon,
                priority,
            });
        }

        // Sort by thread serial
        thread_info_vec.sort_by_key(|t| t.thread_serial);

        eprintln!("Thread info: extracted {} thread names", thread_info_vec.len());
        thread_info_vec
    }

    /// Extract system properties by walking java.lang.System.props -> Properties -> map entries.
    /// Returns Vec<(key, value)> pairs extracted from the live Properties object.
    ///
    /// The Properties graph is followed one object at a time through the object
    /// directory, so no string or array body outlives the object being read.
    pub fn extract_system_properties(&self, hprof_path: &Path) -> Vec<(String, String)> {
        let bodies = match AddrBodies::open(&self.heap, hprof_path) {
            Some(b) => b,
            None => return Vec::new(),
        };
        let id_bytes = bodies.id_bytes;

        // Build class_object_id -> class_idx
        let mut class_obj_to_idx: HashMap<u64, u32> = HashMap::new();
        for cls in &self.heap.classes {
            if cls.class_object_id != 0 {
                class_obj_to_idx.insert(cls.class_object_id, cls.index);
            }
        }

        // Find java.lang.System class index
        let system_idx = match self.heap.class_by_name.get("java/lang/System") {
            Some(&idx) => idx,
            None => return Vec::new(),
        };
        let system_class_obj_id = self.heap.classes[system_idx as usize].class_object_id;

        // Phase 1: Find the System.props static field value from its ClassDump
        let props_obj_addr = bodies
            .class_static_ref(system_class_obj_id, "props", &self.heap.string_table)
            .unwrap_or(0);
        if props_obj_addr == 0 {
            eprintln!("System.props: could not find props static field");
            return Vec::new();
        }

        let graph = PropGraph {
            heap: &self.heap,
            bodies: &bodies,
            class_by_address: &class_obj_to_idx,
        };

        // Phase 2: Walk Properties -> internal map -> entries -> key/value strings
        let mut properties: Vec<(String, String)> = Vec::new();

        // Read the Properties instance
        let (props_class_idx, props_fields) = match graph.relevant(props_obj_addr) {
            Some(data) => data,
            None => {
                eprintln!("System.props: props object at {} not found in instances", props_obj_addr);
                return Vec::new();
            }
        };

        let props_class = &self.heap.classes[props_class_idx as usize];

        // Properties extends Hashtable (JDK 8) or holds a map field pointing at a
        // ConcurrentHashMap (JDK 9+). Try JDK 9+ first.
        let map_addr = find_field_ref(&props_fields, props_class, "map", id_bytes, &class_obj_to_idx, &self.heap.classes);

        if let Some(map_addr) = map_addr {
            // JDK 9+: the map field points to ConcurrentHashMap
            walk_chm_entries(map_addr, &graph, &mut properties);
        } else {
            // JDK 8: Properties extends Hashtable, look for the table field
            walk_hashtable_entries(&props_fields, props_class, &graph, &mut properties);
        }

        if !properties.is_empty() {
            properties.sort_by(|a, b| a.0.cmp(&b.0));
            eprintln!("System.props: extracted {} properties", properties.len());
        }

        properties
    }
}

/// Find a named reference field from an instance, walking the class hierarchy.
fn find_field_ref(
    fields: &[u8],
    class_info: &JavaClassInfo,
    target_name: &str,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> Option<u64> {
    let mut current_class_id: Option<u64> = Some(class_info.class_object_id);
    let mut offset = 0usize;

    while let Some(cid) = current_class_id {
        let cls = if cid == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&cid).and_then(|&idx| all_classes.get(idx as usize))
        };

        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.type_code == 2 {
                    if fd.name.as_ref() == target_name {
                        return read_id_from(fields, offset, id_bytes);
                    }
                    offset += id_bytes;
                } else {
                    offset += field_type_size(fd.type_code, id_bytes);
                }
            }
            current_class_id = if cls.super_class_id != 0 { Some(cls.super_class_id) } else { None };
        } else {
            break;
        }
    }
    None
}

/// Address to body lookups for the Properties walk.
///
/// The one-pass collector this replaces kept every String instance and every
/// byte[]/char[] body in memory. The same filter is applied here, but one object
/// at a time, so the walk holds no payload beyond the object it is reading.
struct PropGraph<'a, 'b> {
    heap: &'a HeapDump,
    bodies: &'b AddrBodies<'a>,
    class_by_address: &'a HashMap<u64, u32>,
}

impl PropGraph<'_, '_> {
    fn id_bytes(&self) -> usize {
        self.bodies.id_bytes
    }

    /// Objects the property walk keeps: the property, map, entry and String
    /// classes, plus ConcurrentHashMap Node and Hashtable Entry arrays.
    fn relevant(&self, addr: u64) -> Option<(u32, Vec<u8>)> {
        let body = self.bodies.get(addr)?;
        let class_info = self.heap.classes.get(body.class_idx as usize)?;
        let name = class_info.name.as_ref();
        let keep = match body.kind {
            DI_KIND_INSTANCE => matches!(
                name,
                "java/util/Properties"
                    | "java/util/concurrent/ConcurrentHashMap"
                    | "java/util/Hashtable"
                    | "java/util/concurrent/ConcurrentHashMap$Node"
                    | "java/util/HashMap$Node"
                    | "java/util/LinkedHashMap$Entry"
                    | "java/util/Hashtable$Entry"
                    | "java/lang/String"
                    | "java/lang/System"
            ),
            DI_KIND_OBJ_ARRAY => {
                name.contains("ConcurrentHashMap$Node") || name.contains("Hashtable$Entry")
            }
            _ => false,
        };
        if !keep {
            return None;
        }
        Some((body.class_idx, body.bytes))
    }

    /// byte[]/char[] payloads with a non-empty body.
    fn prim(&self, addr: u64) -> Option<(Vec<u8>, PrimitiveArrayType)> {
        let body = self.bodies.get(addr)?;
        if body.kind != DI_KIND_PRIM_ARRAY || body.bytes.is_empty() {
            return None;
        }
        let prim_type = PrimitiveArrayType::from_type_code(body.prim_type)?;
        if prim_type != PrimitiveArrayType::Byte && prim_type != PrimitiveArrayType::Char {
            return None;
        }
        Some((body.bytes, prim_type))
    }
}

/// Resolve a String object's content from its class, its value reference and
/// that array's body, each read by address.
fn resolve_string_body(
    bodies: &AddrBodies<'_>,
    str_addr: u64,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> Option<String> {
    let body = bodies.get(str_addr)?;
    let class_info = all_classes.get(body.class_idx as usize)?;
    if class_info.name.as_ref() != "java/lang/String" {
        return None;
    }
    let value_addr = find_field_ref(&body.bytes, class_info, "value", id_bytes, class_by_address, all_classes)?;
    let array = bodies.get(value_addr)?;
    if array.bytes.is_empty() {
        return None;
    }
    let prim_type = PrimitiveArrayType::from_type_code(array.prim_type)?;
    let id_size = if id_bytes == 4 { IdSize::U32 } else { IdSize::U64 };
    let mut prim_bodies: HashMap<u64, (Vec<u8>, PrimitiveArrayType)> = HashMap::new();
    prim_bodies.insert(value_addr, (array.bytes, prim_type));
    extract_string_content_with_body(&body.bytes, class_info, id_size, &prim_bodies)
}

/// Walk ConcurrentHashMap entries starting from the CHM instance.
fn walk_chm_entries(chm_addr: u64, graph: &PropGraph<'_, '_>, properties: &mut Vec<(String, String)>) {
    let id_bytes = graph.id_bytes();
    let (chm_class_idx, chm_fields) = match graph.relevant(chm_addr) {
        Some(d) => d,
        None => return,
    };
    let chm_class = match graph.heap.classes.get(chm_class_idx as usize) {
        Some(c) => c,
        None => return,
    };

    // CHM has a table field holding a Node[] array
    let table_addr = match find_field_ref(&chm_fields, chm_class, "table", id_bytes, graph.class_by_address, &graph.heap.classes) {
        Some(a) if a != 0 => a,
        _ => return,
    };

    let table_data = match graph.relevant(table_addr) {
        Some((_table_class_idx, data)) => data,
        None => return,
    };

    let node_count = table_data.len() / id_bytes;
    for i in 0..node_count {
        let node_addr = read_id_from(&table_data, i * id_bytes, id_bytes).unwrap_or(0);
        if node_addr == 0 { continue; }

        // Walk the linked list at this bucket
        let mut current = node_addr;
        let mut depth = 0;
        while current != 0 && depth < 1000 {
            depth += 1;
            let (node_class_idx, node_fields) = match graph.relevant(current) {
                Some(d) => d,
                None => break,
            };
            let node_class = match graph.heap.classes.get(node_class_idx as usize) {
                Some(c) => c,
                None => break,
            };

            // Node has: key (ref), val (ref), next (ref)
            let key_addr = find_field_ref(&node_fields, node_class, "key", id_bytes, graph.class_by_address, &graph.heap.classes);
            let val_addr = find_field_ref(&node_fields, node_class, "val", id_bytes, graph.class_by_address, &graph.heap.classes);

            if let (Some(ka), Some(va)) = (key_addr, val_addr) {
                let key_str = resolve_string(ka, graph);
                let val_str = resolve_string(va, graph);
                if let (Some(k), Some(v)) = (key_str, val_str) {
                    properties.push((k, v));
                }
            }

            // Follow next pointer
            current = find_field_ref(&node_fields, node_class, "next", id_bytes, graph.class_by_address, &graph.heap.classes).unwrap_or(0);
        }
    }
}

/// Walk Hashtable entries (JDK 8 Properties extends Hashtable).
fn walk_hashtable_entries(
    props_fields: &[u8],
    props_class: &JavaClassInfo,
    graph: &PropGraph<'_, '_>,
    properties: &mut Vec<(String, String)>,
) {
    let id_bytes = graph.id_bytes();

    // Hashtable has a table field pointing to Entry[]
    let table_addr = find_field_ref(props_fields, props_class, "table", id_bytes, graph.class_by_address, &graph.heap.classes);
    let table_addr = match table_addr {
        Some(a) if a != 0 => a,
        _ => return,
    };

    let table_data = match graph.relevant(table_addr) {
        Some((_table_class_idx, data)) => data,
        None => return,
    };

    let entry_count = table_data.len() / id_bytes;
    for i in 0..entry_count {
        let entry_addr = read_id_from(&table_data, i * id_bytes, id_bytes).unwrap_or(0);
        if entry_addr == 0 { continue; }

        // Walk linked list
        let mut current = entry_addr;
        let mut depth = 0;
        while current != 0 && depth < 1000 {
            depth += 1;
            let (entry_class_idx, entry_fields) = match graph.relevant(current) {
                Some(d) => d,
                None => break,
            };
            let entry_class = match graph.heap.classes.get(entry_class_idx as usize) {
                Some(c) => c,
                None => break,
            };

            let key_addr = find_field_ref(&entry_fields, entry_class, "key", id_bytes, graph.class_by_address, &graph.heap.classes);
            let val_addr = find_field_ref(&entry_fields, entry_class, "value", id_bytes, graph.class_by_address, &graph.heap.classes);

            if let (Some(ka), Some(va)) = (key_addr, val_addr) {
                let key_str = resolve_string(ka, graph);
                let val_str = resolve_string(va, graph);
                if let (Some(k), Some(v)) = (key_str, val_str) {
                    properties.push((k, v));
                }
            }

            current = find_field_ref(&entry_fields, entry_class, "next", id_bytes, graph.class_by_address, &graph.heap.classes).unwrap_or(0);
        }
    }
}

/// Resolve a String object at the given address to its content.
fn resolve_string(str_addr: u64, graph: &PropGraph<'_, '_>) -> Option<String> {
    let (class_idx, fields) = graph.relevant(str_addr)?;
    let class_info = graph.heap.classes.get(class_idx as usize)?;

    // Check this is actually a String (or verify by class name)
    if class_info.name.as_ref() != "java/lang/String" {
        return None;
    }

    let id_bytes = graph.id_bytes();
    let value_addr = find_field_ref(&fields, class_info, "value", id_bytes, graph.class_by_address, &graph.heap.classes)?;
    let (body, prim_type) = graph.prim(value_addr)?;
    let id_size = if id_bytes == 4 { IdSize::U32 } else { IdSize::U64 };
    let mut prim_bodies: HashMap<u64, (Vec<u8>, PrimitiveArrayType)> = HashMap::new();
    prim_bodies.insert(value_addr, (body, prim_type));
    extract_string_content_with_body(&fields, class_info, id_size, &prim_bodies)
}

/// Read a boolean field by name, walking the class hierarchy.
fn read_boolean_field(
    fields: &[u8],
    class_info: &JavaClassInfo,
    target_name: &str,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> bool {
    let mut current_class_id: Option<u64> = Some(class_info.class_object_id);
    let mut offset = 0usize;

    while let Some(cid) = current_class_id {
        let cls = if cid == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&cid).and_then(|&idx| all_classes.get(idx as usize))
        };

        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.type_code == 4 {
                    // boolean
                    if fd.name.as_ref() == target_name {
                        return fields.get(offset).copied().unwrap_or(0) != 0;
                    }
                    offset += 1;
                } else {
                    offset += field_type_size(fd.type_code, id_bytes);
                }
            }
            current_class_id = if cls.super_class_id != 0 { Some(cls.super_class_id) } else { None };
        } else {
            break;
        }
    }
    false
}

/// Read an int field by name, walking the class hierarchy.
fn read_int_field(
    fields: &[u8],
    class_info: &JavaClassInfo,
    target_name: &str,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> i32 {
    let mut current_class_id: Option<u64> = Some(class_info.class_object_id);
    let mut offset = 0usize;

    while let Some(cid) = current_class_id {
        let cls = if cid == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&cid).and_then(|&idx| all_classes.get(idx as usize))
        };

        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.type_code == 10 {
                    // int
                    if fd.name.as_ref() == target_name {
                        return read_be_i32(fields, offset).unwrap_or(0);
                    }
                    offset += 4;
                } else {
                    offset += field_type_size(fd.type_code, id_bytes);
                }
            }
            current_class_id = if cls.super_class_id != 0 { Some(cls.super_class_id) } else { None };
        } else {
            break;
        }
    }
    0
}

/// Read all fields of an instance, returning (field_name, FieldValue) pairs.
fn read_all_fields(
    fields: &[u8],
    class_info: &JavaClassInfo,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
    heap: &HeapDump,
) -> Vec<(String, FieldValue)> {
    let mut result = Vec::new();
    let mut current_class_id: Option<u64> = Some(class_info.class_object_id);
    let mut offset = 0usize;

    while let Some(cid) = current_class_id {
        let cls = if cid == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&cid).and_then(|&idx| all_classes.get(idx as usize))
        };

        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                let field_name = fd.name.as_ref().to_string();
                match fd.type_code {
                    2 => {
                        // Object reference
                        let ref_id = read_id_from(fields, offset, id_bytes).unwrap_or(0);
                        if ref_id == 0 {
                            result.push((field_name, FieldValue::Reference(None)));
                        } else {
                            let class_name = heap.class_idx_of(ref_id)
                                .and_then(|idx| all_classes.get(idx as usize))
                                .map(|c| c.name.replace('/', "."));
                            result.push((field_name, FieldValue::Reference(class_name)));
                        }
                        offset += id_bytes;
                    }
                    4 => {
                        let v = fields.get(offset).copied().unwrap_or(0) != 0;
                        result.push((field_name, FieldValue::Boolean(v)));
                        offset += 1;
                    }
                    5 => {
                        let v = if offset + 2 <= fields.len() {
                            u16::from_be_bytes([fields[offset], fields[offset+1]])
                        } else { 0 };
                        result.push((field_name, FieldValue::Char(v)));
                        offset += 2;
                    }
                    6 => {
                        let v = if offset + 4 <= fields.len() {
                            u32::from_be_bytes([fields[offset], fields[offset+1], fields[offset+2], fields[offset+3]])
                        } else { 0 };
                        result.push((field_name, FieldValue::Float(v)));
                        offset += 4;
                    }
                    7 => {
                        let v = if offset + 8 <= fields.len() {
                            u64::from_be_bytes([fields[offset],fields[offset+1],fields[offset+2],fields[offset+3],
                                                fields[offset+4],fields[offset+5],fields[offset+6],fields[offset+7]])
                        } else { 0 };
                        result.push((field_name, FieldValue::Double(v)));
                        offset += 8;
                    }
                    8 => {
                        let v = fields.get(offset).copied().unwrap_or(0) as i8;
                        result.push((field_name, FieldValue::Byte(v)));
                        offset += 1;
                    }
                    9 => {
                        let v = if offset + 2 <= fields.len() {
                            i16::from_be_bytes([fields[offset], fields[offset+1]])
                        } else { 0 };
                        result.push((field_name, FieldValue::Short(v)));
                        offset += 2;
                    }
                    10 => {
                        let v = read_be_i32(fields, offset).unwrap_or(0);
                        result.push((field_name, FieldValue::Int(v)));
                        offset += 4;
                    }
                    11 => {
                        let v = if offset + 8 <= fields.len() {
                            i64::from_be_bytes([fields[offset],fields[offset+1],fields[offset+2],fields[offset+3],
                                                fields[offset+4],fields[offset+5],fields[offset+6],fields[offset+7]])
                        } else { 0 };
                        result.push((field_name, FieldValue::Long(v)));
                        offset += 8;
                    }
                    _ => {
                        offset += field_type_size(fd.type_code, id_bytes);
                    }
                }
            }
            current_class_id = if cls.super_class_id != 0 { Some(cls.super_class_id) } else { None };
        } else {
            break;
        }
    }
    result
}

/// Read elements from an object array as class name references.
fn read_obj_array_elements(
    elements: &[u8],
    id_bytes: usize,
    heap: &HeapDump,
    all_classes: &[JavaClassInfo],
) -> Vec<Option<String>> {
    let count = elements.len() / id_bytes;
    let mut result = Vec::with_capacity(count);
    for i in 0..count.min(20) {
        let addr = read_id_from(elements, i * id_bytes, id_bytes).unwrap_or(0);
        if addr == 0 {
            result.push(None);
        } else {
            let class_name = heap
                .class_idx_of(addr)
                .and_then(|idx| all_classes.get(idx as usize))
                .map(|c| c.name.replace('/', "."));
            result.push(class_name);
        }
    }
    result
}

/// Render a primitive array body as a display string (e.g., "{485, -1, ...}").
fn prim_array_display(body: &[u8], prim_type: PrimitiveArrayType) -> String {
    use std::fmt::Write;
    let width = prim_type_size(prim_type);
    let count = if width == 0 { 0 } else { body.len() / width };
    let mut s = String::with_capacity(256);
    s.push('{');
    for i in 0..count {
        if i > 0 {
            s.push_str(", ");
        }
        if i >= 20 {
            s.push_str("...");
            break;
        }
        let at = i * width;
        match prim_type {
            PrimitiveArrayType::Boolean => {
                s.push_str(if body[at] != 0 { "true" } else { "false" });
            }
            PrimitiveArrayType::Byte => {
                write!(s, "{}", body[at] as i8).unwrap();
            }
            PrimitiveArrayType::Char => {
                let ch = u16::from_be_bytes([body[at], body[at + 1]]);
                if ch >= 0x20 && ch < 0x7f {
                    write!(s, "'{}'", char::from_u32(ch as u32).unwrap_or('?')).unwrap();
                } else {
                    write!(s, "\\u{:04x}", ch).unwrap();
                }
            }
            PrimitiveArrayType::Short => {
                write!(s, "{}", i16::from_be_bytes([body[at], body[at + 1]])).unwrap();
            }
            PrimitiveArrayType::Int => {
                let mut b = [0u8; 4];
                b.copy_from_slice(&body[at..at + 4]);
                write!(s, "{}", i32::from_be_bytes(b)).unwrap();
            }
            PrimitiveArrayType::Long => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&body[at..at + 8]);
                write!(s, "{}", i64::from_be_bytes(b)).unwrap();
            }
            PrimitiveArrayType::Float => {
                let mut b = [0u8; 4];
                b.copy_from_slice(&body[at..at + 4]);
                write!(s, "{}", f32::from_bits(u32::from_be_bytes(b))).unwrap();
            }
            PrimitiveArrayType::Double => {
                let mut b = [0u8; 8];
                b.copy_from_slice(&body[at..at + 8]);
                write!(s, "{}", f64::from_bits(u64::from_be_bytes(b))).unwrap();
            }
        }
    }
    s.push('}');
    s
}

impl LinearScanCoordinator {
    /// Extract local variable details for a specific thread (OOM thread).
    /// Returns Vec<FrameLocals> keyed by frame depth.
    ///
    /// Only the root objects of that thread's frames are read, one body at a time.
    pub fn extract_oom_local_details(
        &self,
        hprof_path: &Path,
        thread_serial: u32,
    ) -> Vec<FrameLocals> {
        let bodies = match AddrBodies::open(&self.heap, hprof_path) {
            Some(b) => b,
            None => return Vec::new(),
        };
        let id_bytes = bodies.id_bytes;

        // Build class_object_id -> class_idx
        let mut class_obj_to_idx: HashMap<u64, u32> = HashMap::new();
        for cls in &self.heap.classes {
            if cls.class_object_id != 0 {
                class_obj_to_idx.insert(cls.class_object_id, cls.index);
            }
        }

        // Collect local variable object addresses grouped by frame_index for this thread
        let mut frame_objects: std::collections::HashMap<u32, Vec<u64>> = std::collections::HashMap::new();
        for root in &self.heap.roots {
            if root.thread_serial == Some(thread_serial) {
                if let Some(fi) = root.frame_index {
                    frame_objects.entry(fi).or_default().push(root.object_id);
                }
            }
        }

        if frame_objects.is_empty() {
            return Vec::new();
        }

        // Build FrameLocals for each frame depth, reading each local by address.
        let mut frame_locals: Vec<FrameLocals> = Vec::new();
        let mut frame_keys: Vec<u32> = frame_objects.keys().copied().collect();
        frame_keys.sort();

        for depth in frame_keys {
            let addrs = &frame_objects[&depth];
            let mut objects = Vec::new();

            for &addr in addrs {
                let body = match bodies.get(addr) {
                    Some(b) => b,
                    None => continue,
                };
                match body.kind {
                    DI_KIND_PRIM_ARRAY => {
                        let prim_type = match PrimitiveArrayType::from_type_code(body.prim_type) {
                            Some(t) => t,
                            None => continue,
                        };
                        let display = prim_array_display(&body.bytes, prim_type);
                        let class_name = self.heap.class_idx_of(addr)
                            .and_then(|idx| self.heap.classes.get(idx as usize))
                            .map(|c| c.name.replace('/', "."))
                            .unwrap_or_else(|| "unknown".to_string());
                        objects.push(LocalVarDetail {
                            class_name,
                            address: addr,
                            fields: Vec::new(),
                            array_elements: None,
                            prim_array_display: Some(display),
                        });
                    }
                    DI_KIND_OBJ_ARRAY => {
                        let class_name = self.heap.class_idx_of(addr)
                            .and_then(|idx| self.heap.classes.get(idx as usize))
                            .map(|c| c.name.replace('/', "."))
                            .unwrap_or_else(|| "unknown".to_string());
                        let elem_class = self.heap.class_idx_of(addr)
                            .and_then(|idx| {
                                // For object arrays, class name is like "java.lang.Object[]"
                                let arr_class = self.heap.classes.get(idx as usize)?;
                                Some(arr_class.name.replace('/', "."))
                            })
                            .unwrap_or_else(|| "Object".to_string());
                        let elem_refs = read_obj_array_elements(&body.bytes, id_bytes, &self.heap, &self.heap.classes);
                        objects.push(LocalVarDetail {
                            class_name,
                            address: addr,
                            fields: Vec::new(),
                            array_elements: Some((elem_class, elem_refs)),
                            prim_array_display: None,
                        });
                    }
                    DI_KIND_INSTANCE => {
                        if (body.class_idx as usize) >= self.heap.classes.len() {
                            continue;
                        }
                        let class_info = &self.heap.classes[body.class_idx as usize];
                        let class_name = class_info.name.replace('/', ".");
                        let field_values = read_all_fields(
                            &body.bytes,
                            class_info,
                            id_bytes,
                            &class_obj_to_idx,
                            &self.heap.classes,
                            &self.heap,
                        );
                        objects.push(LocalVarDetail {
                            class_name,
                            address: addr,
                            fields: field_values,
                            array_elements: None,
                            prim_array_display: None,
                        });
                    }
                    _ => {}
                }
            }

            frame_locals.push(FrameLocals {
                frame_depth: depth,
                objects,
            });
        }

        eprintln!("OOM local details: {} frames with locals", frame_locals.len());
        frame_locals
    }
}

// ============================================================
// Public field-value resolution API (for the report crate)
// ============================================================
//
// The report crate needs per-object field VALUES at render time to emit
// `<t-s>` scalar tags inside reference chains. The BFS `TraversalData`
// only stores the reference graph (parent/parent_edge/field names), not
// values. This entry point re-parses the HPROF once for a given set of
// object addresses and decodes their fields — mirroring the OOM-locals
// path (`extract_oom_local_details`), but as a free function operating
// over an arbitrary address set.

/// Per-object decoded field values, keyed by object address.
pub type FieldValueMap = HashMap<u64, DecodedObject>;

/// One object's decoded representation, ready for `<t-c>/<t-f>/<t-s>` rendering.
#[derive(Debug, Clone)]
pub struct DecodedObject {
    /// `java.lang.String`-style class name (dots, not slashes).
    pub class_name: String,
    /// For instances: `(field_name, FieldValue)` pairs (walks the class hierarchy).
    pub fields: Vec<(String, FieldValue)>,
    /// For object arrays: element class name + element class-name references.
    pub array_elements: Option<(String, Vec<Option<String>>)>,
    /// For primitive arrays: display string like `{485, -1, ...}`.
    pub prim_array_display: Option<String>,
}

/// Resolved string contents for `String` objects keyed by the String object
/// address. Populated by [`resolve_field_values`] so the renderer can turn a
/// `String` reference into a `<t-s>"text"</t-s>` without re-walking the heap.
pub type StringContentMap = HashMap<u64, String>;

/// Result of resolving field values for a set of objects.
#[derive(Debug, Default)]
pub struct ResolvedValues {
    /// Per-object decoded fields/arrays, keyed by object address.
    pub objects: FieldValueMap,
    /// String object address -> resolved text content.
    pub strings: StringContentMap,
    /// Address -> class_idx for every instance seen during the scan
    /// (used to identify String references and other object types).
    pub addr_to_class_idx: HashMap<u64, u32>,
}

/// Re-parse the HPROF once and decode fields for the given object addresses.
///
/// `obj_addrs` may contain instance, object-array, and primitive-array
/// addresses. Any `String` instance referenced by a decoded field whose
/// backing `byte[]`/`char[]` is also in `obj_addrs` has its text content
/// resolved into `strings`.
///
/// Returns an empty [`ResolvedValues`] if the HPROF cannot be opened/parsed.
pub fn resolve_field_values(
    heap: &HeapDump,
    hprof_path: &Path,
    obj_addrs: &std::collections::HashSet<u64>,
) -> ResolvedValues {
    if obj_addrs.is_empty() {
        return ResolvedValues::default();
    }

    let bodies = match AddrBodies::open(heap, hprof_path) {
        Some(b) => b,
        None => return ResolvedValues::default(),
    };
    let id_bytes = bodies.id_bytes;

    // Build class_object_id -> class_idx
    let mut class_obj_to_idx: HashMap<u64, u32> = HashMap::new();
    for cls in &heap.classes {
        if cls.class_object_id != 0 {
            class_obj_to_idx.insert(cls.class_object_id, cls.index);
        }
    }

    let string_class_idx = heap.class_by_name.get("java/lang/String").copied();

    // Read the requested objects only, one body at a time.
    let mut instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();
    let mut obj_arrays: HashMap<u64, Vec<u8>> = HashMap::new();
    let mut prim_arrays: HashMap<u64, (Vec<u8>, PrimitiveArrayType)> = HashMap::new();
    let mut addr_to_class_idx: HashMap<u64, u32> = HashMap::new();

    for &addr in obj_addrs {
        let body = match bodies.get(addr) {
            Some(b) => b,
            None => continue,
        };
        match body.kind {
            DI_KIND_INSTANCE => {
                if (body.class_idx as usize) < heap.classes.len() {
                    instances.insert(addr, (body.class_idx, body.bytes));
                    addr_to_class_idx.insert(addr, body.class_idx);
                }
            }
            DI_KIND_OBJ_ARRAY => {
                obj_arrays.insert(addr, body.bytes);
                addr_to_class_idx.insert(addr, body.class_idx);
            }
            DI_KIND_PRIM_ARRAY => {
                // prim_arrays keeps owned bodies: this pass outlives any single read.
                if !body.bytes.is_empty() {
                    if let Some(prim_type) = PrimitiveArrayType::from_type_code(body.prim_type) {
                        prim_arrays.insert(addr, (body.bytes, prim_type));
                    }
                }
            }
            _ => {}
        }
    }

    // Resolve String contents: every String instance whose backing array is
    // available gets decoded into the strings map.
    let mut strings: StringContentMap = HashMap::new();
    if let Some(str_idx) = string_class_idx {
        for (&addr, (cls_idx, fields)) in &instances {
            if *cls_idx != str_idx {
                continue;
            }
            let cls = &heap.classes[str_idx as usize];
            let id_size_enum = if id_bytes == 4 { IdSize::U32 } else { IdSize::U64 };
            if let Some(text) =
                extract_string_content_with_body(fields, cls, id_size_enum, &prim_arrays)
            {
                strings.insert(addr, text);
            }
        }
    }

    // Decode each object into a DecodedObject.
    let mut objects: FieldValueMap = HashMap::new();

    // Primitive arrays.
    for (addr, (body, _prim_type)) in &prim_arrays {
        let class_name = addr_to_class_idx
            .get(addr)
            .and_then(|&idx| heap.classes.get(idx as usize))
            .map(|c| c.name.replace('/', "."))
            .unwrap_or_else(|| "primitive[]".to_string());
        objects.insert(
            *addr,
            DecodedObject {
                class_name,
                fields: Vec::new(),
                array_elements: None,
                prim_array_display: Some(format!("[{} bytes]", body.len())),
            },
        );
    }

    // Object arrays.
    for (addr, elems) in &obj_arrays {
        let class_name = addr_to_class_idx
            .get(addr)
            .and_then(|&idx| heap.classes.get(idx as usize))
            .map(|c| c.name.replace('/', "."))
            .unwrap_or_else(|| "Object[]".to_string());
        let elem_class = heap
            .class_idx_of(*addr)
            .and_then(|idx| heap.classes.get(idx as usize))
            .map(|c| c.name.replace('/', "."))
            .unwrap_or_else(|| "Object".to_string());
        let elem_refs = read_obj_array_elements(
            elems,
            id_bytes,
            heap,
            &heap.classes,
        );
        objects.insert(
            *addr,
            DecodedObject {
                class_name,
                fields: Vec::new(),
                array_elements: Some((elem_class, elem_refs)),
                prim_array_display: None,
            },
        );
    }

    // Instances.
    for (addr, (class_idx, fields)) in &instances {
        let class_info = &heap.classes[*class_idx as usize];
        let class_name = class_info.name.replace('/', ".");
        let decoded = read_all_fields(
            fields,
            class_info,
            id_bytes,
            &class_obj_to_idx,
            &heap.classes,
            heap,
        );
        objects.insert(
            *addr,
            DecodedObject {
                class_name,
                fields: decoded,
                array_elements: None,
                prim_array_display: None,
            },
        );
    }

    ResolvedValues {
        objects,
        strings,
        addr_to_class_idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- array_shallow_size: must match Java's array getBodySize ----
    // Formula: align8(header_size + 4 [length field] + element_size * length)

    #[test]
    fn array_size_empty_byte_array() {
        // byte[0], header=12: align8(12+4+0) = align8(16) = 16
        assert_eq!(array_shallow_size(0, 1, 12), 16);
    }

    #[test]
    fn array_size_single_element_byte() {
        // byte[1], header=12: align8(12+4+1) = align8(17) = 24
        assert_eq!(array_shallow_size(1, 1, 12), 24);
    }

    #[test]
    fn array_size_object_array_compressed() {
        // Object[4], header=12, refSize=4: align8(12+4+16) = align8(32) = 32
        assert_eq!(array_shallow_size(4, 4, 12), 32);
    }

    #[test]
    fn array_size_large_prim_array() {
        // int[1000], header=12: align8(12+4+4000) = align8(4016) = 4016
        assert_eq!(array_shallow_size(1000, 4, 12), 4016);
    }

    #[test]
    fn array_size_8byte_header() {
        // byte[0] on 32-bit, header=8: align8(8+4+0) = align8(12) = 16
        assert_eq!(array_shallow_size(0, 1, 8), 16);
    }

    // ---- field_type_size: HPROF type codes ----

    #[test]
    fn field_type_object_ref_4byte() {
        assert_eq!(field_type_size(2, 4), 4);
    }

    #[test]
    fn field_type_object_ref_8byte() {
        assert_eq!(field_type_size(2, 8), 8);
    }

    #[test]
    fn field_type_boolean() {
        assert_eq!(field_type_size(4, 8), 1);
    }

    #[test]
    fn field_type_int() {
        assert_eq!(field_type_size(10, 8), 4);
    }

    #[test]
    fn field_type_long() {
        assert_eq!(field_type_size(7, 8), 8);
    }

    // ---- boxed_primitive_size ----

    #[test]
    fn boxed_size_all_types() {
        assert_eq!(boxed_primitive_size("java/lang/Byte"), 1);
        assert_eq!(boxed_primitive_size("java/lang/Boolean"), 1);
        assert_eq!(boxed_primitive_size("java/lang/Short"), 2);
        assert_eq!(boxed_primitive_size("java/lang/Character"), 2);
        assert_eq!(boxed_primitive_size("java/lang/Integer"), 4);
        assert_eq!(boxed_primitive_size("java/lang/Float"), 4);
        assert_eq!(boxed_primitive_size("java/lang/Long"), 8);
        assert_eq!(boxed_primitive_size("java/lang/Double"), 8);
        assert_eq!(boxed_primitive_size("java/lang/String"), 0); // not boxed
    }

    // ---- compute_zero_stats / classify_prim_array ----

    #[test]
    fn zero_stats_detects_trailing_and_high_bits() {
        // int[4], big-endian: {0x00000001, 0, 0, 0}
        let body: Vec<u8> = [
            0, 0, 0, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]
        .to_vec();
        let (trailing, zero_elements, high_bits_zero) = compute_zero_stats(&body, 4);
        assert_eq!(trailing, 3);
        assert_eq!(zero_elements, 3);
        assert_eq!(high_bits_zero, 4);
    }

    #[test]
    fn classify_primitive_array_categories() {
        let config = ScanConfig::default();
        // int[1] -> 1-length, waste is the full body (24 bytes).
        assert_eq!(
            classify_prim_array(1, &[0, 0, 0, 0], 4, &config, 24),
            Some((ss_stats::problem_types::PRIM_ARRAY_LEN_ONE, 24))
        );
        // int[4] all zero -> all-zero, waste is the full body.
        let zeros = vec![0u8; 16];
        assert_eq!(
            classify_prim_array(4, &zeros, 4, &config, 32),
            Some((ss_stats::problem_types::PRIM_ARRAY_ALL_ZERO, 32))
        );
    }

    #[test]
    fn classify_trailing_sparse_and_high_bits() {
        let config = ScanConfig::default();

        // int[8]: four non-zero elements followed by four zeros.
        let mut trailing = Vec::new();
        for _ in 0..4 {
            trailing.extend_from_slice(&[1, 0, 0, 0]);
        }
        trailing.extend(std::iter::repeat(0).take(16));
        assert_eq!(
            classify_prim_array(8, &trailing, 4, &config, 48),
            Some((ss_stats::problem_types::PRIM_ARRAY_TRAIL_ZEROES, 16))
        );

        // int[10]: eight zero elements, then two non-zero elements.
        let mut sparse = vec![0u8; 32];
        sparse.extend_from_slice(&[1, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(
            classify_prim_array(10, &sparse, 4, &config, 56),
            Some((ss_stats::problem_types::PRIM_ARRAY_SPARSE, 32))
        );

        // int[10]: all elements use only the low half (high bits zero).
        let mut high_bits = Vec::new();
        for _ in 0..10 {
            high_bits.extend_from_slice(&[0, 0, 0, 1]);
        }
        assert_eq!(
            classify_prim_array(10, &high_bits, 4, &config, 56),
            Some((ss_stats::problem_types::PRIM_ARRAY_ZERO_HIGH_BITS, 20))
        );
    }
}
