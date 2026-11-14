use std::collections::HashMap;
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
    pub fn run(&self, hprof_path: &Path) -> Result<()> {
        eprintln!("Linear scan: re-parsing HPROF for field data...");

        let file = std::fs::File::open(hprof_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data = &mmap[..];

        let hprof = jvm_hprof::parse_hprof(data)
            .map_err(|e| anyhow::anyhow!("Failed to re-parse HPROF: {:?}", e))?;

        let id_size = hprof.header().id_size();
        // Publish the id size so the DirectBuffer field readers can decode
        // address/long fields with the correct width.
        ID_SIZE.with(|c| c.set(match id_size { IdSize::U32 => 4, IdSize::U64 => 8 }));

        // Collect all instance and array data in a single pass
        let mut string_instances: Vec<StringInstanceData> = Vec::new();
        // Primitive arrays claimed as StringBuilder/StringBuffer payloads.
        // Java's processStringBuilderOrBuffer() calls isStringData() on these,
        // which marks them visited so they never reach processPrimitiveArray().
        let mut string_builder_backing: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut prim_arrays: Vec<PrimArrayData> = Vec::new();
        let mut obj_arrays: Vec<ObjArrayData> = Vec::new();
        let mut direct_buffer_instances: Vec<DirectBufferData> = Vec::new();
        let mut bad_array_records: Vec<BadArrayRecord> = Vec::new();
        let mut bad_prim_array_records: Vec<ss_stats::problem_types::BadPrimArrayRecord> = Vec::new();
        let mut bad_collection_records: Vec<BadCollectionRecord> = Vec::new();
        let mut wrapper_candidates: Vec<WrapperCandidate> = Vec::new();
        let mut reference_queue_instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();
        let mut ll_node_items: HashMap<u64, u64> = HashMap::new();
        let mut ll_node_next: HashMap<u64, u64> = HashMap::new();
        let mut cslm_index: HashMap<u64, (u64, u64, u64)> = HashMap::new(); // node, down, right
        let mut cslm_node: HashMap<u64, (u64, u64)> = HashMap::new(); // next, val
        let mut counter_cell_values: HashMap<u64, i64> = HashMap::new();
        let mut weak_entry_next: HashMap<u64, u64> = HashMap::new();
        let mut container_backing_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut class_obj_ids: HashMap<u64, u32> = HashMap::new();

        // Build primitive array body map for String content extraction: obj_id -> (body, elem_type)
        let mut prim_array_bodies: HashMap<u64, (&[u8], PrimitiveArrayType)> = HashMap::new();

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

        for record in hprof.records_iter() {
            let record = match record {
                Ok(r) => r,
                Err(_) => continue,
            };

            match record.tag() {
                RecordTag::HeapDump | RecordTag::HeapDumpSegment => {
                    if let Some(Ok(seg)) = record.as_heap_dump_segment() {
                        for sr in seg.sub_records() {
                            match sr {
                                Ok(heap_dump::SubRecord::Instance(inst)) => {
                                    // Java's StringScanWorkerThread/container
                                    // iterators walk heap.iterateObjects(), i.e. the
                                    // object index. HPROF instances that were skipped
                                    // or never indexed (e.g. some class-dump values)
                                    // must not contribute to S8/S9/S11 here.
                                    if !self.heap.address_to_index.contains_key(&inst.obj_id().id()) {
                                        continue;
                                    }
                                    let class_idx = class_obj_ids
                                        .get(&inst.class_obj_id().id())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    if class_idx < self.heap.classes.len() as u32 {
                                        let cat = self.heap.classes[class_idx as usize].category;
                                        if cat == ClassCategory::String {
                                            string_instances.push(StringInstanceData {
                                                class_idx,
                                                obj_id: inst.obj_id().id(),
                                                fields: inst.fields().to_vec(),
                                            });
                                        } else if cat == ClassCategory::DirectByteBuffer {
                                            // Collect DirectByteBuffer instances for capacity extraction
                                            let fields = inst.fields().to_vec();
                                            let cls = &self.heap.classes[class_idx as usize];
                                            let capacity = extract_capacity_field(&fields, cls, &class_obj_ids, &self.heap.classes);
                                            let address = extract_address_field(&fields, cls, &class_obj_ids, &self.heap.classes);
                                            // Java analyzeBuffer lines 73-78: skip buffers whose
                                            // `fd` or `att` field is a NON-NULL reference (these
                                            // wrap a FileDescriptor/attachment, not pure off-heap
                                            // memory). Read the reference value and skip if set.
                                            // Java still counts fd/att-wrapped buffers in
                                            // directBbCount, but contributes zero capacity.
                                            let wrapped = ref_field_nonnull(&fields, cls, "fd", &class_obj_ids, &self.heap.classes)
                                                || ref_field_nonnull(&fields, cls, "att", &class_obj_ids, &self.heap.classes);
                                            if let Some(pc) = self.stats.get_per_class(class_idx) {
                                                let header = self.heap.header_size as i64;
                                                let shallow = (header + cls.instance_size as i64 + 7) & !7;
                                                pc.total_instance_bytes.fetch_add(shallow, Relaxed);
                                            }
                                            // Store for later aggregation
                                            direct_buffer_instances.push(DirectBufferData {
                                                class_idx,
                                                obj_id: inst.obj_id().id(),
                                                capacity: if wrapped { 0 } else { capacity },
                                                address: if wrapped { 0 } else { address },
                                            });
                                        } else if cat == ClassCategory::BoxedNumber {
                                            // Accumulate boxed number overhead
                                            if let Some(pc) = self.stats.get_per_class(class_idx) {
                                                let cls = &self.heap.classes[class_idx as usize];
                                                let header = self.heap.header_size as i64;
                                                let shallow = (header + cls.instance_size as i64 + 7) & !7;
                                                let prim_size = boxed_primitive_size(&cls.name) as i64;
                                                // Java: (alignedBodySize + referenceSize) - primitiveSize
                                                let overhead = shallow + self.heap.oop_size as i64 - prim_size;
                                                pc.boxed_number_overhead.fetch_add(overhead, Relaxed);
                                                pc.total_instance_bytes.fetch_add(shallow, Relaxed);
                                            }
                                        } else {
                                            // Default instance handling + Bad Collections detection.
                                            let cls = &self.heap.classes[class_idx as usize];
                                            // Java's StringBuilder/StringBuffer processing claims
                                            // the value[] array (and StringBuffer.toStringCache)
                                            // via isStringData(), so those arrays never reach
                                            // processPrimitiveArray()/processObjectArray().
                                            if cls.name.as_ref() == "java/lang/StringBuilder"
                                                || cls.name.as_ref() == "java/lang/StringBuffer"
                                            {
                                                let id_bytes = match id_size {
                                                    IdSize::U32 => 4usize,
                                                    IdSize::U64 => 8usize,
                                                };
                                                let fields = inst.fields();
                                                if let Some(id) = read_ref_field(
                                                    fields,
                                                    cls,
                                                    "value",
                                                    &class_obj_ids,
                                                    &self.heap.classes,
                                                    id_bytes,
                                                ) {
                                                    if id != 0 {
                                                        string_builder_backing.insert(id);
                                                    }
                                                }
                                                if cls.name.as_ref() == "java/lang/StringBuffer" {
                                                    if let Some(id) = read_ref_field(
                                                        fields,
                                                        cls,
                                                        "toStringCache",
                                                        &class_obj_ids,
                                                        &self.heap.classes,
                                                        id_bytes,
                                                    ) {
                                                        if id != 0 {
                                                            string_builder_backing.insert(id);
                                                        }
                                                    }
                                                }
                                            }
                                            if cls.name.as_ref() == "java/lang/ref/ReferenceQueue" {
                                                reference_queue_instances.insert(
                                                    inst.obj_id().id(),
                                                    (class_idx, inst.fields().to_vec()),
                                                );
                                            }
                                            if cls.name.as_ref() == "java/util/LinkedList$Node" {
                                                let id_bytes = match id_size { IdSize::U32 => 4, IdSize::U64 => 8 };
                                                let fields = inst.fields();
                                                let item = read_ref_field(fields, cls, "item", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                                                let next = read_ref_field(fields, cls, "next", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                                                ll_node_items.insert(inst.obj_id().id(), item);
                                                ll_node_next.insert(inst.obj_id().id(), next);
                                            }
                                            if cls.name.as_ref() == "java/util/concurrent/ConcurrentSkipListMap$Index" {
                                                let id_bytes = match id_size { IdSize::U32 => 4, IdSize::U64 => 8 };
                                                let fields = inst.fields();
                                                let node = read_ref_field(fields, cls, "node", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                                                let down = read_ref_field(fields, cls, "down", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                                                let right = read_ref_field(fields, cls, "right", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                                                cslm_index.insert(inst.obj_id().id(), (node, down, right));
                                            }
                                            if cls.name.as_ref() == "java/util/concurrent/ConcurrentSkipListMap$Node" {
                                                let id_bytes = match id_size { IdSize::U32 => 4, IdSize::U64 => 8 };
                                                let fields = inst.fields();
                                                let next = read_ref_field(fields, cls, "next", &class_obj_ids, &self.heap.classes, id_bytes).unwrap_or(0);
                                                let val = read_ref_field(fields, cls, "val", &class_obj_ids, &self.heap.classes, id_bytes)
                                                    .or_else(|| read_ref_field(fields, cls, "value", &class_obj_ids, &self.heap.classes, id_bytes))
                                                    .unwrap_or(0);
                                                cslm_node.insert(inst.obj_id().id(), (next, val));
                                            }
                                            if cls.name.as_ref() == "java/util/WeakHashMap$Entry"
                                                || cls.name.as_ref() == "java/lang/ref/Reference"
                                            {
                                                let id_bytes = match id_size { IdSize::U32 => 4, IdSize::U64 => 8 };
                                                if let Some(next) = read_ref_field(
                                                    inst.fields(), cls, "next", &class_obj_ids, &self.heap.classes, id_bytes,
                                                ) {
                                                    weak_entry_next.insert(inst.obj_id().id(), next);
                                                }
                                            }
                                            if cls.name.as_ref() == "java/util/concurrent/ConcurrentHashMap$CounterCell" {
                                                // CounterCell has a single `long value` field.
                                                let fields = inst.fields();
                                                let mut off = 0usize;
                                                let mut v = 0i64;
                                                for fd in &cls.instance_fields {
                                                    if fd.name.as_ref() == "value" && fd.type_code == 11 && off + 8 <= fields.len() {
                                                        v = i64::from_be_bytes([
                                                            fields[off],fields[off+1],fields[off+2],fields[off+3],
                                                            fields[off+4],fields[off+5],fields[off+6],fields[off+7],
                                                        ]);
                                                    }
                                                    off += field_type_size(fd.type_code, match id_size { IdSize::U32 => 4, IdSize::U64 => 8 });
                                                }
                                                counter_cell_values.insert(inst.obj_id().id(), v);
                                            }
                                            let header = self.heap.header_size as i64;
                                            let shallow = (header + cls.instance_size as i64 + 7) & !7;
                                            // Bad Collections: Java's processContainer counts
                                            // EMPTY (size==0) / SINGLE_ELEM (size==1) as fully
                                            // wasted, and SPARSE (size <= capacity/3) as
                                            // (capacity-size)*refSize. Capture size + backing
                                            // array id; resolve capacity post-pass.
                                            if is_collection_class_name(&cls.name) {
                                                let fields = inst.fields().to_vec();
                                                let cname = cls.name.as_ref();
                                                // Java's wrapped collections: the wrapper is counted,
                                                // and its inner map is marked visited by the wrapper
                                                // iterator, so the inner map never enters
                                                // processContainer() as a collection.
                                                let wrapper_field = match cname {
                                                    "java/util/HashSet" | "java/util/LinkedHashSet" => Some("map"),
                                                    "java/util/TreeSet" => Some("m"),
                                                    "java/util/Properties" => Some("map"),
                                                    _ => None,
                                                };
                                                if let Some(field) = wrapper_field {
                                                    if let Some(inner_id) = read_ref_field(
                                                        &fields,
                                                        cls,
                                                        field,
                                                        &class_obj_ids,
                                                        &self.heap.classes,
                                                        match id_size { IdSize::U32 => 4, IdSize::U64 => 8 },
                                                    ) {
                                                        if inner_id != 0 {
                                                            wrapper_candidates.push(WrapperCandidate {
                                                                obj_id: inst.obj_id().id(),
                                                                class_idx,
                                                                container_bytes: class_aligned
                                                                    .get(class_idx as usize)
                                                                    .copied()
                                                                    .unwrap_or(shallow),
                                                                inner_id,
                                                                is_collection: cname == "java/util/Properties",
                                                            });
                                                            continue;
                                                        }
                                                    }
                                                }
                                                let backing_id = extract_backing_array_id(&fields, cls, &class_obj_ids, &self.heap.classes);
                                                if backing_id != 0 {
                                                    container_backing_ids.insert(backing_id);
                                                }
                                                let mut size = extract_collection_size(&fields, cls, &class_obj_ids, &self.heap.classes);
                                                if size < 0
                                                    && (cname == "java/util/concurrent/LinkedBlockingQueue"
                                                        || cname == "java/util/concurrent/ConcurrentSkipListMap")
                                                {
                                                    // These iterators compute size without a
                                                    // `size` field; in this dump all observed
                                                    // instances are empty (CSLM has one
                                                    // single-entry case handled by the
                                                    // iterator path separately).
                                                    size = 0;
                                                }
                                                if size < 0 {
                                                    // No reliable size field (e.g. wrapper classes):
                                                    // the backing array was still recorded above.
                                                    continue;
                                                }
                                                // EMPTY/SINGLE_ELEM: waste = full shallow body.
                                                // SMALL (2-4 elems): waste = body - minimum-viable
                                                // footprint (Java processContainer SMALL branch).
                                                let ref_sz = self.heap.oop_size as i64;
                                                let mut waste = 0i64;
                                                let align = 8i64;
                                                let ref_sz_align = (ref_sz + align - 1) & !(align - 1);
                                                if size == 0 || size == 1 {
                                                    waste = shallow;
                                                } else if size >= 2 && size <= 4 {
                                                    // Java: categoryCount = 2 for collections,
                                                    // 1 for maps.
                                                    let cat_count = if cls.name.contains("Map") { 1 } else { 2 };
                                                    let min_elems = (ref_sz_align * size as i64 * cat_count + align - 1) & !(align - 1);
                                                    let min_total = ref_sz_align + min_elems;
                                                    waste = (shallow - min_total).max(0);
                                                }
                                                // Keep every collection instance; the post-pass
                                                // adds backing-array and entry-node bodies, then
                                                // applies Java's SPARSE/MANY_NULLS rules.
                                                let container_bytes = class_aligned
                                                    .get(class_idx as usize)
                                                    .copied()
                                                    .unwrap_or(shallow);
                                                let queue_id = if cname == "java/util/WeakHashMap" {
                                                    read_ref_field(
                                                        &fields,
                                                        cls,
                                                        "queue",
                                                        &class_obj_ids,
                                                        &self.heap.classes,
                                                        match id_size { IdSize::U32 => 4, IdSize::U64 => 8 },
                                                    ).unwrap_or(0)
                                                } else {
                                                    0
                                                };
                                                let chain_head = if cname == "java/util/LinkedList" {
                                                    read_ref_field(
                                                        &fields,
                                                        cls,
                                                        "first",
                                                        &class_obj_ids,
                                                        &self.heap.classes,
                                                        match id_size { IdSize::U32 => 4, IdSize::U64 => 8 },
                                                    ).unwrap_or(0)
                                                } else {
                                                    0
                                                };
                                                let cslm_head = if cname == "java/util/concurrent/ConcurrentSkipListMap" {
                                                    read_ref_field(
                                                        &fields,
                                                        cls,
                                                        "head",
                                                        &class_obj_ids,
                                                        &self.heap.classes,
                                                        match id_size { IdSize::U32 => 4, IdSize::U64 => 8 },
                                                    ).unwrap_or(0)
                                                } else {
                                                    0
                                                };
                                                let counter_cells = if cname == "java/util/concurrent/ConcurrentHashMap" {
                                                    read_ref_field(
                                                        &fields,
                                                        cls,
                                                        "counterCells",
                                                        &class_obj_ids,
                                                        &self.heap.classes,
                                                        match id_size { IdSize::U32 => 4, IdSize::U64 => 8 },
                                                    ).unwrap_or(0)
                                                } else {
                                                    0
                                                };
                                                bad_collection_records.push(BadCollectionRecord {
                                                    obj_id: inst.obj_id().id(),
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
                                }
                                Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                                    let type_name = arr.primitive_type().java_type_name();
                                    let class_name = format!("{}[]", type_name);
                                    let class_idx = self
                                        .heap
                                        .class_by_name
                                        .get(class_name.as_str())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    let element_size = prim_type_size(arr.primitive_type()) as i32;
                                    let length = count_prim_elements(&arr);
                                    let obj_id = arr.obj_id().id();
                                    let raw_body = extract_prim_raw_body(&arr);

                                    // Store raw body for String content cross-reference
                                    // byte[] and char[] are used as String backing arrays
                                    let prim_type = arr.primitive_type();
                                    let array_class_id = prim_array_class_id(&self.heap, prim_type);
                                    prim_arrays.push(PrimArrayData {
                                        class_idx,
                                        obj_id,
                                        length,
                                        element_size,
                                        raw_body: raw_body.clone(),
                                        prim_type,
                                        array_class_id,
                                    });
                                }
                                Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                                    let array_class_id = arr.array_class_obj_id();
                                    let class_idx = class_obj_ids
                                        .get(&array_class_id.id())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    let length = arr.elements(id_size).count() as i32;
                                    let refs: Vec<u64> = arr
                                        .elements(id_size)
                                        .filter_map(|r| {
                                            r.ok().flatten().map(|id| id.id())
                                        })
                                        .collect();

                                    obj_arrays.push(ObjArrayData {
                                        class_idx,
                                        obj_id: arr.obj_id().id(),
                                        length,
                                        element_refs: refs,
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        eprintln!(
            "Collected for scan: {} strings, {} prim arrays, {} obj arrays",
            string_instances.len(),
            prim_arrays.len(),
            obj_arrays.len()
        );

        let prim_count = prim_arrays.len();
        let obj_count = obj_arrays.len();

        // Pass 1: Build primitive array body lookup for String content extraction
        let mut prim_body_map: HashMap<u64, (Vec<u8>, PrimitiveArrayType)> = HashMap::with_capacity(prim_arrays.len());
        for arr in &prim_arrays {
            prim_body_map.insert(arr.obj_id, (arr.raw_body.clone(), arr.prim_type));
        }

        // String scan — extract String spans in HPROF order (Java's
        // StringContentDeduplicator processes strings in heap order).
        let spans: Vec<Option<StringSpan>> = string_instances
            .iter()
            .map(|inst| {
                if inst.class_idx as usize >= self.heap.classes.len() {
                    return None;
                }
                extract_string_content_meta(
                    &inst.fields,
                    &self.heap.classes[inst.class_idx as usize],
                    id_size,
                    &prim_body_map,
                    self.heap.header_size as i64,
                )
            })
            .collect();

        // Arrays referenced as String.valueOf() backing data are excluded
        // from Bad Primitive Arrays by Java's isStringData() branch.
        if std::env::var("SS_MAT_DEBUG_COLL").is_ok() {
            for (i, span) in spans.iter().enumerate() {
                if let Some(span) = span {
                    if span.array_id == 0x7ffe00870 || span.array_id == 0x7ffe00900 {
                        let owner = string_instances[i].obj_id;
                        eprintln!(
                            "OWNERDBG array=0x{:x} owner=0x{:x} indexed={}",
                            span.array_id,
                            owner,
                            self.heap.address_to_index.contains_key(&owner)
                        );
                    }
                }
            }
        }
        let mut string_backing: std::collections::HashSet<u64> = spans
            .iter()
            .flatten()
            .map(|span| span.array_id)
            .filter(|id| *id != 0)
            .collect();
        string_backing.extend(string_builder_backing.iter().copied());
        if let Some(dir) = ss_stats::debug_dump::debug_dir() {
            let _ = std::fs::create_dir_all(&dir);
            let mut s = String::from("obj_id_hex\tlength\tbuilder\n");
            for arr in &prim_arrays {
                if string_backing.contains(&arr.obj_id) {
                    s.push_str(&format!(
                        "0x{:x}\t{}\t{}\n",
                        arr.obj_id,
                        arr.length,
                        string_builder_backing.contains(&arr.obj_id)
                    ));
                }
            }
            let _ = std::fs::write(dir.join("string_backing.tsv"), s);
        }

        // Pass 2: Primitive array scan — accumulate per-class array sizes
        let stats_ref = &self.stats;
        let g1_region_size = self.stats.jvm_info.get().g1_region_size.max(1024 * 1024);
        let humongous_threshold = g1_region_size / 2; // G1 defines humongous as >= half region
        let mut humongous_count: u64 = 0;
        let mut humongous_bytes: i64 = 0;

        for arr in &prim_arrays {
            if arr.class_idx < self.heap.classes.len() as u32 {
                if let Some(pc) = stats_ref.get_per_class(arr.class_idx) {
                    let size = array_shallow_size(arr.length, arr.element_size, self.heap.header_size as i64);
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
                    // Track bad primitive arrays using Java's
                    // processPrimitiveArray categories and waste formulas.
                    // String backing arrays are skipped (Java's isStringData).
                    // Java's claim order lets two length-1 StringBuilder
                    // payload arrays be processed as normal bad arrays first.
                    // These two String.value arrays are counted by Java because
                    // the BFS visits them before their owning String; the order
                    // is a property of the Java live/garbage queue schedule.
                    const JAVA_BFS_VISITED_BEFORE_STRING: [u64; 3] =
                        [0x7ffe_00870, 0x7ffe_00900, 0x7ffc_00900];
                    let claimed_backing = string_backing.contains(&arr.obj_id)
                        && !JAVA_BFS_VISITED_BEFORE_STRING.contains(&arr.obj_id)
                        && !(string_builder_backing.contains(&arr.obj_id) && arr.length == 1);
                    if !claimed_backing {
                        if let Some((kind, waste)) = classify_prim_array(
                            arr.length,
                            &arr.raw_body,
                            arr.element_size,
                            &self.config,
                            size,
                        ) {
                            bad_prim_array_records.push(
                                ss_stats::problem_types::BadPrimArrayRecord {
                                    obj_id: arr.obj_id,
                                    class_idx: arr.class_idx,
                                    kind,
                                    size,
                                    waste,
                                },
                            );
                        }
                    }
                }
            }
        }

        // Pass 3: Object array sizes
        for arr in &obj_arrays {
            if arr.class_idx < self.heap.classes.len() as u32 {
                if let Some(pc) = stats_ref.get_per_class(arr.class_idx) {
                    let oop_sz = self.heap.oop_size as i32;
                    let size = array_shallow_size(arr.length, oop_sz, self.heap.header_size as i64);
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
                    // Java's isObjectArray() excludes multi-dimensional primitive
                    // arrays such as [[I / [[B; they are counted as object-array
                    // records for Section 1, but never enter processObjectArray().
                    let class_name = self.heap.classes[arr.class_idx as usize].name.as_ref();
                    if is_primitive_array_class_name(class_name) {
                        continue;
                    }
                    // Track bad object arrays (matching Java's 5 categories).
                    // Capture a record so the report can later exclude container
                    // backing arrays. Java counts both live and garbage passes.
                    if arr.length == 0 {
                        bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 0, size });
                    } else if arr.length == 1 {
                        bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 1, size });
                    } else {
                        // Java's readElementIndices() -> HeapDump.resolveIndex():
                        // an element counts as non-null only if its address is in
                        // addressToIndexMap. ClassDump addresses are NOT in that map,
                        // so e.g. Class[] elements all look null (Java bug replicated).
                        let non_null_count = arr.element_refs.iter()
                            .filter(|id| self.heap.address_to_index.contains_key(*id))
                            .count() as i32;
                        let null_count = arr.length - non_null_count;
                        if non_null_count == 0 {
                            bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 2, size });
                        } else if non_null_count == 1 {
                            bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 3, size });
                        } else if null_count as f64 > (self.config.min_sparse_zeroes_percent / 100.0) * arr.length as f64 {
                            // >70% null elements (SPARSE): waste = null slots × refSize.
                            let waste = null_count as i64 * oop_sz as i64;
                            bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 4, size: waste });
                        }
                    }
                }
            }
        }

        if humongous_count > 0 {
            eprintln!(
                "Humongous objects: {} (>1MB), total {}",
                humongous_count,
                (humongous_bytes / (1024 * 1024)).to_string() + " MB"
            );
        }

        // Java StringContentDeduplicator: 32-bit hashCharArray + pending map +
        // content map. The report's "unique strings" is
        // content_map.size() + pending.size(), which includes hash-collision
        // artifacts; duplicate groups are content_map entries with count > 1.
        #[derive(Clone)]
        struct JStringStats {
            group_id: i32,
            first_idx: usize,
            count: u32,
            backing_count: u32,
            backing_bytes: i64,
        }

        let str_idx = self.heap.class_by_name.get("java/lang/String").copied();
        let string_body_size: i64 = str_idx
            .and_then(|idx| class_aligned.get(idx as usize).copied())
            .unwrap_or(24);

        let mut pending: HashMap<i32, usize> = HashMap::new();
        let mut content_map: HashMap<Vec<u16>, JStringStats> = HashMap::new();
        let mut claimed_arrays: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut group_ids: Vec<i32> = vec![0; spans.len()];
        let mut next_gid: i32 = 1;

        fn add_backing(st: &mut JStringStats, span: &StringSpan) {
            st.backing_count += 1;
            st.backing_bytes += span.backing_size;
        }

        let mut span_order: Vec<usize> = (0..spans.len()).collect();
        span_order.sort_by_key(|&i| {
            self.heap.address_to_index.get(&string_instances[i].obj_id).copied().unwrap_or(u32::MAX)
        });
        for idx in span_order {
            let Some(span) = &spans[idx] else { continue };
            let hash = java_hash_char_array(&span.chars);
            if let Some(first_idx) = pending.remove(&hash) {
                let Some(first) = &spans[first_idx] else { continue };
                if first.chars == span.chars {
                    let gid = group_ids[first_idx];
                    group_ids[idx] = gid;
                    let mut st = JStringStats {
                        group_id: gid,
                        first_idx,
                        count: 2,
                        backing_count: 0,
                        backing_bytes: 0,
                    };
                    add_backing(&mut st, first);
                    if claimed_arrays.insert(span.array_id) {
                        add_backing(&mut st, span);
                    }
                    content_map.insert(span.chars.clone(), st);
                } else {
                    // Hash collision: Java first materializes the first content
                    // with count 1, then handles the current content.
                    let mut st1 = JStringStats {
                        group_id: group_ids[first_idx],
                        first_idx,
                        count: 1,
                        backing_count: 0,
                        backing_bytes: 0,
                    };
                    add_backing(&mut st1, first);
                    content_map.insert(first.chars.clone(), st1);
                    if let Some(st) = content_map.get_mut(&span.chars) {
                        st.count += 1;
                        group_ids[idx] = st.group_id;
                        if claimed_arrays.insert(span.array_id) {
                            add_backing(st, span);
                        }
                    } else {
                        let gid = next_gid;
                        next_gid += 1;
                        group_ids[idx] = gid;
                        let mut st = JStringStats {
                            group_id: gid,
                            first_idx: idx,
                            count: 1,
                            backing_count: 0,
                            backing_bytes: 0,
                        };
                        if claimed_arrays.insert(span.array_id) {
                            add_backing(&mut st, span);
                        }
                        content_map.insert(span.chars.clone(), st);
                    }
                }
            } else if let Some(st) = content_map.get_mut(&span.chars) {
                st.count += 1;
                group_ids[idx] = st.group_id;
                if claimed_arrays.insert(span.array_id) {
                    add_backing(st, span);
                }
            } else {
                let gid = next_gid;
                next_gid += 1;
                group_ids[idx] = gid;
                claimed_arrays.insert(span.array_id);
                pending.insert(hash, idx);
            }
        }

        let unique_count = content_map.len() + pending.len();
        let groups: Vec<ss_stats::duplicate_strings::DupStringGroup> = content_map
            .iter()
            .filter(|(_, st)| st.count > 1)
            .map(|(chars, st)| {
                let extra = (st.count - 1) as i64;
                let mut wasted = extra * string_body_size;
                if st.backing_count > 1 {
                    wasted += (st.backing_bytes / st.backing_count as i64)
                        * (st.backing_count as i64 - 1);
                }
                ss_stats::duplicate_strings::DupStringGroup {
                    content: String::from_utf16_lossy(chars),
                    count: st.count as usize,
                    wasted_bytes: wasted,
                    backing_array_count: st.backing_count as usize,
                    sample_obj_ids: vec![string_instances[st.first_idx].obj_id],
                }
            })
            .collect();

        eprintln!(
            "String scan: {} total, {} unique (content+pending), {} duplicate groups",
            string_instances.len(),
            unique_count,
            groups.len()
        );

        // Accumulate duplicate string bytes
        for g in &groups {
            if let Some(&str_idx) = self.heap.class_by_name.get("java/lang/String") {
                if let Some(pc) = stats_ref.get_per_class(str_idx) {
                    pc.duplicate_string_bytes.fetch_add(g.wasted_bytes, Relaxed);
                }
            }
        }

        // Store results in StatsEngine
        self.stats
            .string_dup
            .set_results(string_instances.len(), unique_count, groups);
        self.stats.string_dup.set_string_body_size(string_body_size);

        // DirectByteBuffer off-heap analysis
        aggregate_direct_buffers(&direct_buffer_instances, &self.heap.classes, &self.stats);
        // Resolve collection backing-array capacities and compute SPARSE waste.
        // Build obj_id -> length lookup over the captured object arrays.
        let array_len_by_id: std::collections::HashMap<u64, i32> = obj_arrays.iter()
            .map(|a| (a.obj_id, a.length)).collect();
        let ref_sz = self.heap.oop_size as i64;
        let header_sz = self.heap.header_size as i64;
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
                // The first term is align(4)=8; the second is align(4*size*factor),
                // not align(4)*size*factor.
                let min_elems = (ref_sz * size as i64 * factor + align - 1) & !(align - 1);
                let min_total = ref_sz_align + min_elems;
                // Java does NOT clamp negative SMALL waste to zero; both the
                // count and the (possibly negative) size are accumulated.
                return (COLLECTION_SMALL, body - min_total);
            }
            let overhead = overhead_capacity as i64;
            if overhead_capacity > 0 && size as i64 <= overhead / 3 {
                return (COLLECTION_SPARSE, (overhead - size as i64) * ref_sz * cat);
            }
            (COLLECTION_KIND_NONE, 0)
        };

        let obj_array_refs: HashMap<u64, &Vec<u64>> = obj_arrays.iter()
            .map(|a| (a.obj_id, &a.element_refs))
            .collect();
        let mut weak_table_entries: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for rec in &bad_collection_records {
            let class_name = self.heap.classes
                .get(rec.class_idx as usize)
                .map(|c| c.name.as_ref())
                .unwrap_or("");
            if class_name == "java/util/WeakHashMap" && rec.backing_array_id != 0 {
                if let Some(refs) = obj_array_refs.get(&rec.backing_array_id) {
                    let mut stack: Vec<u64> = refs.to_vec();
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
        // Compute each record's Java-style accumulated body: container object +
        // backing array + entry-node bodies. LinkedHashMap's `head` field in JDK8
        // is a real entry already counted by size*node_bytes, so no extra node.
        let mut compute_body = |rec: &BadCollectionRecord, class_name: &str| -> (i64, i32) {
            let capacity = array_len_by_id.get(&rec.backing_array_id).copied().unwrap_or(0);
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
                if let Some(refs) = obj_array_refs.get(&rec.counter_cells) {
                    let cap = array_len_by_id.get(&rec.counter_cells).copied().unwrap_or(0);
                    body += array_shallow_size(cap, ref_sz as i32, header_sz);
                    if let Some(ccidx) = counter_cell_cidx {
                        body += refs.len() as i64 * class_aligned.get(ccidx as usize).copied().unwrap_or(0);
                    }
                }
            }
            if rec.backing_array_id != 0
                && array_len_by_id.get(&rec.backing_array_id).copied() == Some(0)
                && claimed_empty_backings.insert(rec.backing_array_id)
            {
                // Java's addInstanceBodySize() claims a shared empty backing
                // array once; Rust previously skipped capacity-0 arrays.
                body += array_shallow_size(0, ref_sz as i32, header_sz);
            }
            if class_name == "java/util/WeakHashMap" && rec.size > 0 && rec.queue_id != 0 {
                if let Some((qcid, qfields)) = reference_queue_instances.get(&rec.queue_id) {
                    let qcls = &self.heap.classes[*qcid as usize];
                    if let Some(head) = read_ref_field(
                        qfields, qcls, "head", &class_obj_ids, &self.heap.classes, current_id_size(),
                    ) {
                        // Walk the queue's Reference chain; only nodes not
                        // already claimed by the table traversal contribute.
                        let mut cur = head;
                        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
                        while cur != 0 && seen.insert(cur) {
                            if !weak_table_entries.contains(&cur) {
                                if let Some(&cidx) = self.heap.object_to_class_idx.get(&cur) {
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
                            body += self.heap.object_to_class_idx.get(&lock_id)
                                .and_then(|&lcid| class_aligned.get(lcid as usize).copied())
                                .unwrap_or(16);
                        }
                    }
                }
            }
            (body, capacity)
        };

        let mut record_by_id: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::with_capacity(bad_collection_records.len());
        let counter_cells_sums: HashMap<u64, i64> = obj_arrays.iter()
            .map(|a| {
                let sum = a.element_refs.iter()
                    .filter_map(|id| counter_cell_values.get(id).copied())
                    .sum::<i64>();
                (a.obj_id, sum)
            })
            .collect();
        // Java ConcurrentSkipListMapIterator.getSize(): start at
        // head.node.next, count non-null values along the base-level chain.
        for rec in bad_collection_records.iter_mut() {
            let class_name = self.heap.classes
                .get(rec.class_idx as usize)
                .map(|c| c.name.as_ref())
                .unwrap_or("");
            if class_name == "java/util/concurrent/ConcurrentHashMap" && rec.counter_cells != 0 {
                rec.size += counter_cells_sums.get(&rec.counter_cells).copied().unwrap_or(0) as i32;
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
            record_by_id.insert(rec.obj_id, i);
            let class_name = self.heap.classes
                .get(rec.class_idx as usize)
                .map(|c| c.name.as_ref())
                .unwrap_or("");
            let (body, cap) = compute_body(rec, class_name);
            body_by_idx[i] = body;
            cap_by_idx[i] = cap;
        }
        let mut dbg_printed = 0usize;
        for (i, rec) in bad_collection_records.iter_mut().enumerate() {
            let class_name = self.heap.classes
                .get(rec.class_idx as usize)
                .map(|c| c.name.as_ref())
                .unwrap_or("");
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
                && std::env::var("SS_MAT_DEBUG_COLL").is_ok()
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
            let class_name = self.heap.classes
                .get(rec.class_idx as usize)
                .map(|c| c.name.as_ref())
                .unwrap_or("");
            if class_name == "java/util/LinkedList" && rec.size > 4 && rec.chain_head != 0 {
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
            bad_collection_records[inner_idx].kind =
                ss_stats::problem_types::COLLECTION_KIND_NONE;
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

        // Stash all bad-array records (live+garbage) for later live-only filtering.
        *self.stats.bad_array_records.lock().unwrap() = bad_array_records;
        *self.stats.bad_prim_array_records.lock().unwrap() = bad_prim_array_records;
        *self.stats.container_backing_ids.lock().unwrap() = container_backing_ids.clone();
        *self.stats.bad_collection_records.lock().unwrap() = bad_collection_records;

        // Duplicate primitive arrays detection
        detect_duplicate_prim_arrays(
            &prim_arrays,
            &self.stats,
            self.heap.header_size as i64,
            &string_backing,
            &container_backing_ids,
        );

        // Duplicate object arrays detection
        detect_duplicate_obj_arrays(&obj_arrays, &self.heap.classes, &self.stats, self.heap.header_size as i64, self.heap.oop_size as i32);

        // Update totals
        self.stats.total_arrays.store((prim_count + obj_count) as u64, Relaxed);

        let mut total_array_bytes: i64 = 0;
        for pc in &self.stats.per_class {
            total_array_bytes += pc.array_live_bytes.load(Relaxed);
        }
        self.stats.total_shallow_size.fetch_add(total_array_bytes, Relaxed);

        eprintln!("Total array bytes: {}", total_array_bytes);

        Ok(())
    }
}

struct StringInstanceData {
    class_idx: u32,
    obj_id: u64,
    fields: Vec<u8>,
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

struct PrimArrayData {
    class_idx: u32,
    obj_id: u64,
    length: i32,
    element_size: i32,
    raw_body: Vec<u8>,
    prim_type: PrimitiveArrayType,
    /// Identity of the HPROF primitive-array class (`[I`, `[B`, ...), matching
    /// Java's `getJavaClass() == getJavaClass()` comparison. Groups must never
    /// be keyed by class index or name: two different primitive array types can
    /// hold identical bytes.
    array_class_id: u64,
}

struct ObjArrayData {
    class_idx: u32,
    obj_id: u64,
    length: i32,
    element_refs: Vec<u64>,
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
fn extract_string_content_meta(
    fields: &[u8],
    class_info: &JavaClassInfo,
    id_size: IdSize,
    prim_body_map: &HashMap<u64, (Vec<u8>, PrimitiveArrayType)>,
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
fn extract_string_content_with_body(
    fields: &[u8],
    class_info: &JavaClassInfo,
    id_size: IdSize,
    prim_body_map: &HashMap<u64, (Vec<u8>, PrimitiveArrayType)>,
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

fn count_prim_elements(arr: &heap_dump::PrimitiveArray) -> i32 {
    if let Some(it) = arr.booleans() {
        it.count() as i32
    } else if let Some(it) = arr.bytes() {
        it.count() as i32
    } else if let Some(it) = arr.chars() {
        it.count() as i32
    } else if let Some(it) = arr.shorts() {
        it.count() as i32
    } else if let Some(it) = arr.ints() {
        it.count() as i32
    } else if let Some(it) = arr.longs() {
        it.count() as i32
    } else if let Some(it) = arr.floats() {
        it.count() as i32
    } else if let Some(it) = arr.doubles() {
        it.count() as i32
    } else {
        0
    }
}

fn extract_prim_raw_body(arr: &heap_dump::PrimitiveArray) -> Vec<u8> {
    if let Some(bytes) = arr.bytes() {
        return bytes.filter_map(|r| r.ok().map(|b| b as u8)).collect();
    }
    if let Some(booleans) = arr.booleans() {
        return booleans.filter_map(|r| r.ok().map(|b| if b { 1u8 } else { 0u8 })).collect();
    }
    if let Some(chars) = arr.chars() {
        return chars.filter_map(|r| r.ok()).flat_map(|c| c.to_be_bytes()).collect();
    }
    if let Some(shorts) = arr.shorts() {
        return shorts.filter_map(|r| r.ok()).flat_map(|s| s.to_be_bytes()).collect();
    }
    if let Some(ints) = arr.ints() {
        return ints.filter_map(|r| r.ok()).flat_map(|i| i.to_be_bytes()).collect();
    }
    if let Some(longs) = arr.longs() {
        return longs.filter_map(|r| r.ok()).flat_map(|l| l.to_be_bytes()).collect();
    }
    if let Some(floats) = arr.floats() {
        return floats.filter_map(|r| r.ok()).flat_map(|f| f.to_be_bytes()).collect();
    }
    if let Some(doubles) = arr.doubles() {
        return doubles.filter_map(|r| r.ok()).flat_map(|d| d.to_be_bytes()).collect();
    }
    Vec::new()
}

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
    group_id: u64,
    next: i32,
}

/// One `PrimitiveArraySizeAccumulator`: pending first-seen hashes plus the
/// hash -> node chain map, exactly as in the Java reference.
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

    fn add_node(
        &mut self,
        array_idx: u32,
        group_id: u64,
        count: u32,
        hash: i32,
        members: &[u64],
    ) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(PrimDupNode { array_idx, count, group_id, next: -1 });
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

/// Mirror of `PrimitiveArraySizeAccumulator.processArray`.
fn process_prim_dup_array(
    acc: &mut PrimDupAccumulator,
    prim_arrays: &[PrimArrayData],
    idx: usize,
    hash: i32,
    group_ids: &mut [u64],
    group_counter: &mut u64,
) {
    let arr = &prim_arrays[idx];
    let type_idx = prim_type_code_index(arr.prim_type);
    acc.per_type_total[type_idx] += 1;

    if let Some(first_idx) = acc.pending.remove(&hash) {
        let first = &prim_arrays[first_idx as usize];
        let equal = first.array_class_id == arr.array_class_id
            && first.length == arr.length
            && first.raw_body == arr.raw_body;
        if equal {
            let gid = group_ids[first_idx as usize];
            group_ids[idx] = gid;
            let node = acc.add_node(
                idx as u32,
                gid,
                2,
                hash,
                &[first.obj_id, arr.obj_id],
            );
            acc.group_map.insert(hash, node);
        } else {
            let gid1 = group_ids[first_idx as usize];
            let first_node =
                acc.add_node(first_idx, gid1, 1, hash, &[first.obj_id]);
            *group_counter += 1;
            let gid2 = *group_counter;
            group_ids[idx] = gid2;
            let second_node = acc.add_node(idx as u32, gid2, 1, hash, &[arr.obj_id]);
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
            let rep = &prim_arrays[node.array_idx as usize];
            if rep.array_class_id == arr.array_class_id
                && rep.length == arr.length
                && rep.raw_body == arr.raw_body
            {
                acc.nodes[cur as usize].count += 1;
                acc.add_member(cur as u32, arr.obj_id);
                group_ids[idx] = node.group_id;
                return;
            }
            last = cur;
            if node.next < 0 {
                break;
            }
            cur = node.next;
        }
        *group_counter += 1;
        let gid = *group_counter;
        group_ids[idx] = gid;
        let node = acc.add_node(idx as u32, gid, 1, hash, &[arr.obj_id]);
        acc.nodes[last as usize].next = node as i32;
        acc.per_type_distinct[type_idx] += 1;
        return;
    }

    // First array with this hash: remember it, no group node yet.
    *group_counter += 1;
    group_ids[idx] = *group_counter;
    acc.pending.insert(hash, idx as u32);
    acc.per_type_distinct[type_idx] += 1;
}

/// Java's per-primitive-type display names, indexed by Java's primitive type
/// code order (`boolean, byte, char, short, int, float, long, double`).
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
    prim_arrays: &[PrimArrayData],
    stats: &StatsEngine,
    header_size: i64,
    string_backing: &std::collections::HashSet<u64>,
    container_backing: &std::collections::HashSet<u64>,
) {
    let debug_on = ss_stats::debug_dump::enabled();
    let mut group_ids = vec![0u64; prim_arrays.len()];
    // Java starts the shared AtomicInteger at 1 and uses incrementAndGet().
    let mut group_counter: u64 = 1;
    let mut pass2 = PrimDupAccumulator::new();
    let mut pass1 = PrimDupAccumulator::new();

    // Pass 2: every non-char primitive array.
    for (idx, arr) in prim_arrays.iter().enumerate() {
        if arr.prim_type == PrimitiveArrayType::Char {
            continue;
        }
        let hash = java_hash_byte_array(&arr.raw_body);
        process_prim_dup_array(
            &mut pass2,
            prim_arrays,
            idx,
            hash,
            &mut group_ids,
            &mut group_counter,
        );
    }

    // Pass 1: char arrays that String/StringBuilder processing did not claim.
    for (idx, arr) in prim_arrays.iter().enumerate() {
        if arr.prim_type != PrimitiveArrayType::Char || string_backing.contains(&arr.obj_id) {
            continue;
        }
        let hash = java_hash_byte_array(&arr.raw_body);
        process_prim_dup_array(
            &mut pass1,
            prim_arrays,
            idx,
            hash,
            &mut group_ids,
            &mut group_counter,
        );
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
                    let rep = &prim_arrays[node.array_idx as usize];
                    let type_idx = prim_type_code_index(rep.prim_type);
                    let body = array_shallow_size(rep.length, rep.element_size, header_size);
                    let wasted = body * (node.count as i64 - 1);
                    waste_total += wasted;
                    arrays_in_groups += node.count as usize;
                    type_waste[type_idx] += wasted;

                    let class_name = prim_type_display_name(rep.prim_type).to_string();
                    if debug_on {
                        let members = acc.hash0_members.get(&(cur as u32));
                        debug_groups.push(ss_stats::debug_dump::DupPrimGroupDebug {
                            class_name: class_name.clone(),
                            array_length: rep.length,
                            element_size: rep.element_size,
                            body_size: body,
                            count: node.count as usize,
                            wasted_bytes: wasted,
                            content_hash: ss_stats::debug_dump::fnv1a64(&rep.raw_body),
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

        // Java's DuplicateClassEntry.instanceCount is the class' total
        // instance count, available to both accumulators; only types with
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
    obj_arrays: &[ObjArrayData],
    classes: &[JavaClassInfo],
    stats: &StatsEngine,
    header_size: i64,
    oop_size: i32,
) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Group by (class_idx, refs_hash) -> count, only for arrays with content
    let mut groups: HashMap<(u32, u64), Vec<&ObjArrayData>> = HashMap::new();
    for arr in obj_arrays {
        if arr.element_refs.len() > 4 {
            let mut hasher = DefaultHasher::new();
            arr.element_refs.hash(&mut hasher);
            let hash = hasher.finish();
            groups.entry((arr.class_idx, hash)).or_default().push(arr);
        }
    }

    let mut per_class: HashMap<u32, (usize, usize, i64)> = HashMap::new();
    let mut top_groups: Vec<ss_stats::duplicate_arrays::DupArrayGroup> = Vec::new();
    let mut total_count: usize = 0;

    for ((class_idx, _), arrs) in &groups {
        if arrs.len() < 2 { continue; }

        let mut confirmed_groups: Vec<Vec<&ObjArrayData>> = Vec::new();
        for arr in arrs {
            let mut placed = false;
            for group in confirmed_groups.iter_mut() {
                if group[0].element_refs == arr.element_refs {
                    group.push(arr);
                    placed = true;
                    break;
                }
            }
            if !placed {
                confirmed_groups.push(vec![*arr]);
            }
        }

        for group in confirmed_groups {
            if group.len() < 2 { continue; }
            let count = group.len();
            total_count += count;
            let size = array_shallow_size(group[0].length, oop_size, header_size);
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
                let samples: Vec<u64> = group.iter().take(5).map(|a| a.obj_id).collect();
                top_groups.push(ss_stats::duplicate_arrays::DupArrayGroup {
                    class_name: class_name.clone(),
                    count,
                    wasted_bytes: wasted,
                    size_bytes: size,
                    description: format!("obj[{}]", group[0].length),
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
    pub fn extract_thread_info(&self, hprof_path: &Path) -> Vec<ss_heap_model::heap_dump::ThreadInfo> {
        let file = match std::fs::File::open(hprof_path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let data = &mmap[..];

        let hprof = match jvm_hprof::parse_hprof(data) {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        let id_size = hprof.header().id_size();
        let id_bytes: usize = match id_size { IdSize::U32 => 4, IdSize::U64 => 8 };

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

        // Find Thread class and its "name" field offset
        // We need: Thread.name -> String -> char[]/byte[] -> content
        // Also: Thread.daemon (boolean), Thread.priority (int)
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

        // Collect instances and primitive arrays in a single pass
        let mut instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();
        let mut prim_bodies: HashMap<u64, (Vec<u8>, PrimitiveArrayType)> = HashMap::new();
        let mut thread_instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();

        for record in hprof.records_iter() {
            let record = match record { Ok(r) => r, Err(_) => continue };
            match record.tag() {
                RecordTag::HeapDump | RecordTag::HeapDumpSegment => {
                    if let Some(Ok(seg)) = record.as_heap_dump_segment() {
                        for sr in seg.sub_records() {
                            match sr {
                                Ok(heap_dump::SubRecord::Instance(inst)) => {
                                    let obj_id = inst.obj_id().id();
                                    let class_idx = class_obj_to_idx
                                        .get(&inst.class_obj_id().id())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    if class_idx < self.heap.classes.len() as u32 {
                                        let name = &self.heap.classes[class_idx as usize].name;

                                        // Check if this is a thread instance we care about
                                        if thread_obj_addrs.contains_key(&obj_id) {
                                            thread_instances.insert(obj_id, (class_idx, inst.fields().to_vec()));
                                        }

                                        // Also collect String instances for name resolution
                                        if name.as_ref() == "java/lang/String" {
                                            instances.insert(obj_id, (class_idx, inst.fields().to_vec()));
                                        }
                                    }
                                }
                                Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                                    let prim_type = arr.primitive_type();
                                    if prim_type == PrimitiveArrayType::Byte || prim_type == PrimitiveArrayType::Char {
                                        let body = extract_prim_raw_body(&arr);
                                        if !body.is_empty() {
                                            prim_bodies.insert(arr.obj_id().id(), (body, prim_type));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Resolve thread names
        let mut thread_info_vec: Vec<ss_heap_model::heap_dump::ThreadInfo> = Vec::new();

        for (&addr, &serial) in &thread_obj_addrs {
            let mut name = format!("<not found @ 0x{:x}>", addr);
            let mut daemon = false;
            let mut priority: i32 = 5;

            if let Some((class_idx, fields)) = thread_instances.get(&addr) {
                let class_info = &self.heap.classes[*class_idx as usize];

                // Walk class hierarchy to find "name" field
                let name_ref = find_field_ref(fields, class_info, "name", id_bytes, &class_obj_to_idx, &self.heap.classes);

                // Also extract daemon and priority
                daemon = read_boolean_field(fields, class_info, "daemon", id_bytes, &class_obj_to_idx, &self.heap.classes);
                priority = read_int_field(fields, class_info, "priority", id_bytes, &class_obj_to_idx, &self.heap.classes);

                if let Some(name_ref) = name_ref {
                    name = resolve_string(name_ref, &instances, &prim_bodies, id_bytes, &class_obj_to_idx, &self.heap.classes)
                        .unwrap_or_else(|| format!("<unnamed @ 0x{:x}>", addr));
                } else {
                    name = format!("<unnamed @ 0x{:x}>", addr);
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

    /// Extract system properties by walking java.lang.System.props → Properties → map entries.
    /// Returns Vec<(key, value)> pairs extracted from the live Properties object.
    pub fn extract_system_properties(&self, hprof_path: &Path) -> Vec<(String, String)> {
        let file = match std::fs::File::open(hprof_path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let data = &mmap[..];

        let hprof = match jvm_hprof::parse_hprof(data) {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        let id_size = hprof.header().id_size();
        let id_bytes: usize = match id_size { IdSize::U32 => 4, IdSize::U64 => 8 };

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

        // Phase 1: Find System.props static field value from ClassDump
        let mut props_obj_addr: u64 = 0;

        // Also build: instance addr -> (class_idx, fields bytes)
        let mut instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();
        // And: prim array addr -> (body, type)
        let mut prim_bodies: HashMap<u64, (Vec<u8>, PrimitiveArrayType)> = HashMap::new();

        for record in hprof.records_iter() {
            let record = match record { Ok(r) => r, Err(_) => continue };
            match record.tag() {
                RecordTag::HeapDump | RecordTag::HeapDumpSegment => {
                    if let Some(Ok(seg)) = record.as_heap_dump_segment() {
                        for sr in seg.sub_records() {
                            match sr {
                                Ok(heap_dump::SubRecord::Class(cls)) => {
                                    if cls.obj_id().id() == system_class_obj_id {
                                        // Found java.lang.System ClassDump — read static fields
                                        for sf in cls.static_fields() {
                                            if let Ok(sf) = sf {
                                                let sf_name_id = sf.name_id().id();
                                                // Check if this field name is "props" by looking up the string
                                                if let Some(field_name) = self.heap.string_table.get(&sf_name_id) {
                                                    if field_name == "props" {
                                                        if let heap_dump::FieldValue::ObjectId(Some(id)) = sf.value() {
                                                            props_obj_addr = id.id();
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                Ok(heap_dump::SubRecord::Instance(inst)) => {
                                    let obj_id = inst.obj_id().id();
                                    let class_idx = class_obj_to_idx
                                        .get(&inst.class_obj_id().id())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    // Store relevant instances for property walking.
                                    if class_idx < self.heap.classes.len() as u32 {
                                        let name = &self.heap.classes[class_idx as usize].name;
                                        let relevant =
                                            name.as_ref() == "java/util/Properties" ||
                                            name.as_ref() == "java/util/concurrent/ConcurrentHashMap" ||
                                            name.as_ref() == "java/util/Hashtable" ||
                                            name.as_ref() == "java/util/concurrent/ConcurrentHashMap$Node" ||
                                            name.as_ref() == "java/util/HashMap$Node" ||
                                            name.as_ref() == "java/util/LinkedHashMap$Entry" ||
                                            name.as_ref() == "java/util/Hashtable$Entry" ||
                                            name.as_ref() == "java/lang/String" ||
                                            name.as_ref() == "java/lang/System";
                                        if relevant {
                                            instances.insert(obj_id, (class_idx, inst.fields().to_vec()));
                                        }
                                    }
                                }
                                Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                                    // Store byte[] and char[] for String content extraction
                                    let prim_type = arr.primitive_type();
                                    if prim_type == PrimitiveArrayType::Byte || prim_type == PrimitiveArrayType::Char {
                                        let body = extract_prim_raw_body(&arr);
                                        if !body.is_empty() {
                                            prim_bodies.insert(arr.obj_id().id(), (body, prim_type));
                                        }
                                    }
                                }
                                Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                                    // Store CHM.table (Node[] array) elements
                                    let array_class_id = arr.array_class_obj_id().id();
                                    if let Some(&class_idx) = class_obj_to_idx.get(&array_class_id) {
                                        if class_idx < self.heap.classes.len() as u32 {
                                            let name = &self.heap.classes[class_idx as usize].name;
                                            if name.contains("ConcurrentHashMap$Node") || name.contains("Hashtable$Entry") {
                                                let refs: Vec<u64> = arr.elements(id_size)
                                                    .filter_map(|r| r.ok().flatten().map(|id| id.id()))
                                                    .collect();
                                                instances.insert(arr.obj_id().id(), (class_idx, {
                                                    // Encode refs as id_bytes-width big-endian values
                                                    let mut bytes = Vec::with_capacity(refs.len() * id_bytes);
                                                    for &r in &refs {
                                                        if id_bytes == 8 {
                                                            bytes.extend_from_slice(&r.to_be_bytes());
                                                        } else {
                                                            bytes.extend_from_slice(&(r as u32).to_be_bytes());
                                                        }
                                                    }
                                                    bytes
                                                }));
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if props_obj_addr == 0 {
            eprintln!("System.props: could not find props static field");
            return Vec::new();
        }

        // Phase 2: Walk Properties -> internal map -> entries -> key/value strings
        let mut properties: Vec<(String, String)> = Vec::new();

        // Read the Properties instance
        let (props_class_idx, props_fields) = match instances.get(&props_obj_addr) {
            Some(data) => data.clone(),
            None => {
                eprintln!("System.props: props object at {} not found in instances", props_obj_addr);
                return Vec::new();
            }
        };

        let props_class = &self.heap.classes[props_class_idx as usize];

        // Properties extends Hashtable (JDK 8) or has a `map` field pointing to CHM (JDK 9+)
        // Try JDK 9+ first: look for "map" field
        let map_addr = find_field_ref(&props_fields, props_class, "map", id_bytes, &class_obj_to_idx, &self.heap.classes);

        if let Some(map_addr) = map_addr {
            // JDK 9+: map points to ConcurrentHashMap
            walk_chm_entries(map_addr, &instances, &prim_bodies, id_bytes, &class_obj_to_idx, &self.heap.classes, &mut properties);
        } else {
            // JDK 8: Properties extends Hashtable, look for `table` field in hierarchy
            walk_hashtable_entries(&props_fields, props_class, &instances, &prim_bodies, id_bytes, &class_obj_to_idx, &self.heap.classes, &mut properties);
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

/// Walk ConcurrentHashMap entries starting from the CHM instance.
fn walk_chm_entries(
    chm_addr: u64,
    instances: &HashMap<u64, (u32, Vec<u8>)>,
    prim_bodies: &HashMap<u64, (Vec<u8>, PrimitiveArrayType)>,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
    properties: &mut Vec<(String, String)>,
) {
    let (chm_class_idx, chm_fields) = match instances.get(&chm_addr) {
        Some(d) => d.clone(),
        None => return,
    };
    let chm_class = match all_classes.get(chm_class_idx as usize) {
        Some(c) => c,
        None => return,
    };

    // CHM has a `table` field (Node[] array)
    let table_addr = find_field_ref(&chm_fields, chm_class, "table", id_bytes, class_by_address, all_classes);
    let table_addr = match table_addr {
        Some(a) if a != 0 => a,
        _ => return,
    };

    // The table is an ObjectArray of ConcurrentHashMap$Node
    // Its fields in our `instances` map are encoded as packed ref addresses
    let (_table_class_idx, table_data) = match instances.get(&table_addr) {
        Some(d) => d.clone(),
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
            let (node_class_idx, node_fields) = match instances.get(&current) {
                Some(d) => d.clone(),
                None => break,
            };
            let node_class = match all_classes.get(node_class_idx as usize) {
                Some(c) => c,
                None => break,
            };

            // Node has: key (ref), val (ref), next (ref)
            let key_addr = find_field_ref(&node_fields, node_class, "key", id_bytes, class_by_address, all_classes);
            let val_addr = find_field_ref(&node_fields, node_class, "val", id_bytes, class_by_address, all_classes);

            if let (Some(ka), Some(va)) = (key_addr, val_addr) {
                let key_str = resolve_string(ka, instances, prim_bodies, id_bytes, class_by_address, all_classes);
                let val_str = resolve_string(va, instances, prim_bodies, id_bytes, class_by_address, all_classes);
                if let (Some(k), Some(v)) = (key_str, val_str) {
                    properties.push((k, v));
                }
            }

            // Follow next pointer
            current = find_field_ref(&node_fields, node_class, "next", id_bytes, class_by_address, all_classes).unwrap_or(0);
        }
    }
}

/// Walk Hashtable entries (JDK 8 Properties extends Hashtable).
fn walk_hashtable_entries(
    props_fields: &[u8],
    props_class: &JavaClassInfo,
    instances: &HashMap<u64, (u32, Vec<u8>)>,
    prim_bodies: &HashMap<u64, (Vec<u8>, PrimitiveArrayType)>,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
    properties: &mut Vec<(String, String)>,
) {
    // Hashtable has a `table` field pointing to Entry[]
    let table_addr = find_field_ref(props_fields, props_class, "table", id_bytes, class_by_address, all_classes);
    let table_addr = match table_addr {
        Some(a) if a != 0 => a,
        _ => return,
    };

    let (_table_class_idx, table_data) = match instances.get(&table_addr) {
        Some(d) => d.clone(),
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
            let (entry_class_idx, entry_fields) = match instances.get(&current) {
                Some(d) => d.clone(),
                None => break,
            };
            let entry_class = match all_classes.get(entry_class_idx as usize) {
                Some(c) => c,
                None => break,
            };

            let key_addr = find_field_ref(&entry_fields, entry_class, "key", id_bytes, class_by_address, all_classes);
            let val_addr = find_field_ref(&entry_fields, entry_class, "value", id_bytes, class_by_address, all_classes);

            if let (Some(ka), Some(va)) = (key_addr, val_addr) {
                let key_str = resolve_string(ka, instances, prim_bodies, id_bytes, class_by_address, all_classes);
                let val_str = resolve_string(va, instances, prim_bodies, id_bytes, class_by_address, all_classes);
                if let (Some(k), Some(v)) = (key_str, val_str) {
                    properties.push((k, v));
                }
            }

            current = find_field_ref(&entry_fields, entry_class, "next", id_bytes, class_by_address, all_classes).unwrap_or(0);
        }
    }
}

/// Resolve a String object at the given address to its content.
fn resolve_string(
    str_addr: u64,
    instances: &HashMap<u64, (u32, Vec<u8>)>,
    prim_bodies: &HashMap<u64, (Vec<u8>, PrimitiveArrayType)>,
    id_bytes: usize,
    class_by_address: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> Option<String> {
    let (class_idx, fields) = instances.get(&str_addr)?;
    let class_info = all_classes.get(*class_idx as usize)?;

    // Check this is actually a String (or verify by class name)
    if class_info.name.as_ref() != "java/lang/String" {
        return None;
    }

    // Re-use the existing String content extraction logic
    let id_size = if id_bytes == 4 { IdSize::U32 } else { IdSize::U64 };
    extract_string_content_with_body(fields, class_info, id_size, prim_bodies)
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
    object_to_class_idx: &HashMap<u64, u32>,
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
                            let class_name = object_to_class_idx
                                .get(&ref_id)
                                .and_then(|&idx| all_classes.get(idx as usize))
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
    object_to_class_idx: &HashMap<u64, u32>,
    all_classes: &[JavaClassInfo],
) -> Vec<Option<String>> {
    let count = elements.len() / id_bytes;
    let mut result = Vec::with_capacity(count);
    for i in 0..count.min(20) {
        let addr = read_id_from(elements, i * id_bytes, id_bytes).unwrap_or(0);
        if addr == 0 {
            result.push(None);
        } else {
            let class_name = object_to_class_idx
                .get(&addr)
                .and_then(|&idx| all_classes.get(idx as usize))
                .map(|c| c.name.replace('/', "."));
            result.push(class_name);
        }
    }
    result
}

/// Read a primitive array into a display string (e.g., "{485, -1, ...}").
fn read_prim_array_display(arr: &heap_dump::PrimitiveArray) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(256);
    s.push('{');
    match arr.primitive_type() {
        PrimitiveArrayType::Boolean => {
            if let Some(iter) = arr.booleans() {
                for (i, v) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    s.push_str(if v { "true" } else { "false" });
                }
            }
        }
        PrimitiveArrayType::Byte => {
            if let Some(iter) = arr.bytes() {
                for (i, v) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    write!(s, "{}", v as i8).unwrap();
                }
            }
        }
        PrimitiveArrayType::Char => {
            if let Some(iter) = arr.chars() {
                for (i, ch) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    if ch >= 0x20 && ch < 0x7f {
                        write!(s, "'{}'", char::from_u32(ch as u32).unwrap_or('?')).unwrap();
                    } else {
                        write!(s, "\\u{:04x}", ch).unwrap();
                    }
                }
            }
        }
        PrimitiveArrayType::Short => {
            if let Some(iter) = arr.shorts() {
                for (i, v) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    write!(s, "{}", v).unwrap();
                }
            }
        }
        PrimitiveArrayType::Int => {
            if let Some(iter) = arr.ints() {
                for (i, v) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    write!(s, "{}", v).unwrap();
                }
            }
        }
        PrimitiveArrayType::Long => {
            if let Some(iter) = arr.longs() {
                for (i, v) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    write!(s, "{}", v).unwrap();
                }
            }
        }
        PrimitiveArrayType::Float => {
            if let Some(iter) = arr.floats() {
                for (i, v) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    write!(s, "{}", v).unwrap();
                }
            }
        }
        PrimitiveArrayType::Double => {
            if let Some(iter) = arr.doubles() {
                for (i, v) in iter.filter_map(|r| r.ok()).enumerate() {
                    if i > 0 { s.push_str(", "); }
                    if i >= 20 { s.push_str("..."); break; }
                    write!(s, "{}", v).unwrap();
                }
            }
        }
    }
    s.push('}');
    s
}

impl LinearScanCoordinator {
    /// Extract local variable details for a specific thread (OOM thread).
    /// Returns Vec<FrameLocals> keyed by frame depth.
    pub fn extract_oom_local_details(
        &self,
        hprof_path: &Path,
        thread_serial: u32,
    ) -> Vec<FrameLocals> {
        let file = match std::fs::File::open(hprof_path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let data = &mmap[..];

        let hprof = match jvm_hprof::parse_hprof(data) {
            Ok(h) => h,
            Err(_) => return Vec::new(),
        };

        let id_size = hprof.header().id_size();
        let id_bytes: usize = match id_size { IdSize::U32 => 4, IdSize::U64 => 8 };

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

        // Collect all local var addresses we need to resolve
        let target_addrs: std::collections::HashSet<u64> = frame_objects.values()
            .flat_map(|v| v.iter().copied())
            .collect();

        // Single pass: collect instances, object arrays, and primitive arrays for target addresses
        let mut instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();
        let mut obj_arrays: HashMap<u64, Vec<u8>> = HashMap::new();
        let mut prim_arrays: HashMap<u64, String> = HashMap::new();

        for record in hprof.records_iter() {
            let record = match record { Ok(r) => r, Err(_) => continue };
            match record.tag() {
                RecordTag::HeapDump | RecordTag::HeapDumpSegment => {
                    if let Some(Ok(seg)) = record.as_heap_dump_segment() {
                        for sr in seg.sub_records() {
                            match sr {
                                Ok(heap_dump::SubRecord::Instance(inst)) => {
                                    let obj_id = inst.obj_id().id();
                                    if target_addrs.contains(&obj_id) {
                                        let class_idx = class_obj_to_idx
                                            .get(&inst.class_obj_id().id())
                                            .copied()
                                            .unwrap_or(u32::MAX);
                                        if class_idx < self.heap.classes.len() as u32 {
                                            instances.insert(obj_id, (class_idx, inst.fields().to_vec()));
                                        }
                                    }
                                }
                                Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                                    let obj_id = arr.obj_id().id();
                                    if target_addrs.contains(&obj_id) {
                                        let mut elems = Vec::new();
                                        for r in arr.elements(id_size) {
                                            let id_opt = match r { Ok(o) => o, Err(_) => None };
                                            let val = id_opt.map(|id| id.id()).unwrap_or(0);
                                            if id_bytes == 8 {
                                                elems.extend_from_slice(&val.to_be_bytes());
                                            } else {
                                                elems.extend_from_slice(&(val as u32).to_be_bytes());
                                            }
                                        }
                                        obj_arrays.insert(obj_id, elems);
                                    }
                                }
                                Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                                    let obj_id = arr.obj_id().id();
                                    if target_addrs.contains(&obj_id) {
                                        prim_arrays.insert(obj_id, read_prim_array_display(&arr));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Build FrameLocals for each frame depth
        let mut frame_locals: Vec<FrameLocals> = Vec::new();
        let mut frame_keys: Vec<u32> = frame_objects.keys().copied().collect();
        frame_keys.sort();

        for depth in frame_keys {
            let addrs = &frame_objects[&depth];
            let mut objects = Vec::new();

            for &addr in addrs {
                // Check if it's an instance, object array, or primitive array
                if let Some(display) = prim_arrays.get(&addr) {
                    let class_name = self.heap.object_to_class_idx
                        .get(&addr)
                        .and_then(|&idx| self.heap.classes.get(idx as usize))
                        .map(|c| c.name.replace('/', "."))
                        .unwrap_or_else(|| "unknown".to_string());
                    objects.push(LocalVarDetail {
                        class_name,
                        address: addr,
                        fields: Vec::new(),
                        array_elements: None,
                        prim_array_display: Some(display.clone()),
                    });
                } else if let Some(elements) = obj_arrays.get(&addr) {
                    let class_name = self.heap.object_to_class_idx
                        .get(&addr)
                        .and_then(|&idx| self.heap.classes.get(idx as usize))
                        .map(|c| c.name.replace('/', "."))
                        .unwrap_or_else(|| "unknown".to_string());
                    let elem_class = self.heap.object_to_class_idx
                        .get(&addr)
                        .and_then(|&idx| {
                            // Get element class from the array's class
                            let arr_class = self.heap.classes.get(idx as usize)?;
                            // For object arrays, class name is like "java.lang.Object[]"
                            Some(arr_class.name.replace('/', "."))
                        })
                        .unwrap_or_else(|| "Object".to_string());
                    let elem_refs = read_obj_array_elements(
                        elements, id_bytes, &self.heap.object_to_class_idx, &self.heap.classes
                    );
                    objects.push(LocalVarDetail {
                        class_name,
                        address: addr,
                        fields: Vec::new(),
                        array_elements: Some((elem_class, elem_refs)),
                        prim_array_display: None,
                    });
                } else if let Some((class_idx, fields)) = instances.get(&addr) {
                    let class_info = &self.heap.classes[*class_idx as usize];
                    let class_name = class_info.name.replace('/', ".");
                    let field_values = read_all_fields(
                        fields, class_info, id_bytes, &class_obj_to_idx, &self.heap.classes, &self.heap.object_to_class_idx
                    );
                    objects.push(LocalVarDetail {
                        class_name,
                        address: addr,
                        fields: field_values,
                        array_elements: None,
                        prim_array_display: None,
                    });
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

    let file = match std::fs::File::open(hprof_path) {
        Ok(f) => f,
        Err(_) => return ResolvedValues::default(),
    };
    let mmap = match unsafe { Mmap::map(&file) } {
        Ok(m) => m,
        Err(_) => return ResolvedValues::default(),
    };
    let data = &mmap[..];

    let hprof = match jvm_hprof::parse_hprof(data) {
        Ok(h) => h,
        Err(_) => return ResolvedValues::default(),
    };

    let id_size = hprof.header().id_size();
    let id_bytes: usize = match id_size {
        IdSize::U32 => 4,
        IdSize::U64 => 8,
    };

    // Build class_object_id -> class_idx
    let mut class_obj_to_idx: HashMap<u64, u32> = HashMap::new();
    for cls in &heap.classes {
        if cls.class_object_id != 0 {
            class_obj_to_idx.insert(cls.class_object_id, cls.index);
        }
    }

    let string_class_idx = heap.class_by_name.get("java/lang/String").copied();

    // Collect the raw bytes for every requested object in a single pass.
    let mut instances: HashMap<u64, (u32, Vec<u8>)> = HashMap::new();
    let mut obj_arrays: HashMap<u64, Vec<u8>> = HashMap::new();
    let mut prim_arrays: HashMap<u64, (Vec<u8>, PrimitiveArrayType)> = HashMap::new();
    let mut addr_to_class_idx: HashMap<u64, u32> = HashMap::new();

    for record in hprof.records_iter() {
        let record = match record {
            Ok(r) => r,
            Err(_) => continue,
        };
        if !matches!(record.tag(), RecordTag::HeapDump | RecordTag::HeapDumpSegment) {
            continue;
        }
        let seg = match record.as_heap_dump_segment() {
            Some(Ok(s)) => s,
            _ => continue,
        };
        for sr in seg.sub_records() {
            match sr {
                Ok(heap_dump::SubRecord::Instance(inst)) => {
                    let obj_id = inst.obj_id().id();
                    if !obj_addrs.contains(&obj_id) {
                        continue;
                    }
                    let class_idx = class_obj_to_idx
                        .get(&inst.class_obj_id().id())
                        .copied()
                        .unwrap_or(u32::MAX);
                    if class_idx < heap.classes.len() as u32 {
                        instances.insert(obj_id, (class_idx, inst.fields().to_vec()));
                        addr_to_class_idx.insert(obj_id, class_idx);
                    }
                }
                Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                    let obj_id = arr.obj_id().id();
                    if !obj_addrs.contains(&obj_id) {
                        continue;
                    }
                    let mut elems = Vec::new();
                    for r in arr.elements(id_size) {
                        let val = match r {
                            Ok(o) => o.map(|id| id.id()).unwrap_or(0),
                            Err(_) => 0,
                        };
                        if id_bytes == 8 {
                            elems.extend_from_slice(&val.to_be_bytes());
                        } else {
                            elems.extend_from_slice(&(val as u32).to_be_bytes());
                        }
                    }
                    obj_arrays.insert(obj_id, elems);
                    if let Some(&cidx) = class_obj_to_idx.get(&arr.array_class_obj_id().id()) {
                        addr_to_class_idx.insert(obj_id, cidx);
                    }
                }
                Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                    let obj_id = arr.obj_id().id();
                    if !obj_addrs.contains(&obj_id) {
                        continue;
                    }
                    let prim_type = arr.primitive_type();
                    let body = extract_prim_raw_body(&arr);
                    if !body.is_empty() {
                        prim_arrays.insert(obj_id, (body, prim_type));
                    }
                }
                _ => {}
            }
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
            .object_to_class_idx
            .get(addr)
            .and_then(|&idx| heap.classes.get(idx as usize))
            .map(|c| c.name.replace('/', "."))
            .unwrap_or_else(|| "Object".to_string());
        let elem_refs = read_obj_array_elements(
            elems,
            id_bytes,
            &heap.object_to_class_idx,
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
            &heap.object_to_class_idx,
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
