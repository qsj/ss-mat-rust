//! Phase-1 HPROF parse.
//!
//! The dump is never kept resident. Top-level record headers are located with a
//! forward pread scan, the small metadata records (UTF8, LOAD_CLASS, STACK_FRAME,
//! STACK_TRACE) are gathered into one bounded buffer and decoded by jvm-hprof, and
//! every HEAP_DUMP / HEAP_DUMP_SEGMENT record is mapped on its own and walked by
//! the strict raw walker in record_reader. The pages behind each segment are
//! handed back to the kernel as the walk leaves them, so file residency stays near
//! one window instead of the dump size.
//!
//! The same walk feeds the on-disk object directory, so the dump is traversed
//! exactly once here.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use jvm_hprof::{IdSize, RecordTag};
use memmap2::{MmapOptions, UncheckedAdvice};

use ss_heap_model::class_info::{ClassCategory, FieldDesc, JavaClassInfo};
use ss_heap_model::heap_dump::HeapDump;
use ss_heap_model::root::{HeapRootReference, RootType};

use crate::heap_info::HeapInfo;
use crate::objdir_build::{KIND_CLASS_DUMP, ObjDirConfig, ObjDirSink};
use crate::record_reader::{
    self, ClassRec, Fields, HeapItem, ObjRec as WalkRec, RegionOrigin, RootRec, KIND_INSTANCE,
    KIND_OBJ_ARRAY, KIND_PRIM_ARRAY, TAG_HEAP_DUMP, TAG_HEAP_DUMP_SEGMENT,
};

/// HPROF file magic, up to and including the trailing dot of the version.
const MAGIC: &[u8] = b"JAVA PROFILE 1.0.";

/// Bytes of file kept resident behind the heap-dump walk cursor.
const RELEASE_WINDOW: usize = 16 << 20;

/// Bytes scanned per refill while locating top-level record headers.
const SCAN_BUFFER: usize = 1 << 20;

/// Basic type code -> the synthetic array class name the parser creates for it.
const PRIM_ARRAY_NAMES: [&str; 12] = [
    "", "", "", "", "boolean[]", "char[]", "float[]", "double[]", "byte[]", "short[]", "int[]",
    "long[]",
];

/// Parse an HPROF file with the default temporary-directory configuration.
pub fn parse_hprof(path: &Path) -> Result<HeapDump> {
    parse_hprof_with_config(path, &ObjDirConfig::default())
}

/// Parse an HPROF file and produce a HeapDump plus its on-disk object directory.
pub fn parse_hprof_with_config(path: &Path, cfg: &ObjDirConfig) -> Result<HeapDump> {
    let file = File::open(path).with_context(|| format!("Failed to open: {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("Failed to stat: {}", path.display()))?
        .len();

    let header = read_header(&file, file_len)?;
    let id_size_byte = header.id_size;
    let id_size_enum = if id_size_byte == 4 { IdSize::U32 } else { IdSize::U64 };

    eprintln!("HPROF parsed: id_size={}", id_size_byte);

    let mut builder = HeapDumpBuilder::new();
    builder.timestamp_millis = header.timestamp_millis;
    let mut sink = ObjDirSink::new(&cfg.temp_dir)?;

    // The synthetic prefix every small-record batch starts with, so jvm-hprof sees a
    // well-formed file header for each batch it decodes.
    let prefix = header.bytes.clone();
    let mut small: Vec<u8> = Vec::with_capacity(prefix.len() + (1 << 20));
    small.extend_from_slice(&prefix);

    let mut walk = record_reader::WalkStats::default();
    let mut reader = ScanReader::new(file, file_len);
    let mut pos = header.len as u64;
    let mut segments = 0u64;
    let started = std::time::Instant::now();
    while pos < file_len {
        let head = reader.header(pos)?;
        let tag = head[0];
        let body_len = u32::from_be_bytes([head[5], head[6], head[7], head[8]]) as u64;
        let body_off = pos + 9;
        let end = body_off
            .checked_add(body_len)
            .ok_or_else(|| anyhow!("record tag {tag:#04x} at offset {pos}: body length overflows"))?;
        if end > file_len {
            bail!(
                "record tag {tag:#04x} at offset {pos}: body of {body_len} bytes runs past the end of file ({file_len})"
            );
        }
        if tag == TAG_HEAP_DUMP || tag == TAG_HEAP_DUMP_SEGMENT {
            // Records before this one must already be visible to the heap walk
            // (class names, field names, load classes, stack frames).
            flush_small(&mut small, &prefix, id_size_enum, &mut builder)?;
            let stats = walk_segment(
                reader.file(),
                tag,
                pos,
                body_off,
                body_len,
                id_size_byte,
                &mut builder,
                &mut sink,
            )?;
            walk.instances += stats.instances;
            walk.object_arrays += stats.object_arrays;
            walk.prim_arrays += stats.prim_arrays;
            walk.class_dumps += stats.class_dumps;
            walk.gc_roots += stats.gc_roots;
            segments += 1;
        } else {
            small.extend_from_slice(&head);
            reader.append(body_off, body_len, &mut small)?;
        }
        pos = end;
    }
    flush_small(&mut small, &prefix, id_size_enum, &mut builder)?;
    let walk_elapsed = started.elapsed();
    drop(small);
    drop(reader);

    let records = sink.records();
    eprintln!(
        "  Object directory: {} records ({} instances, {} prim arrays, {} obj arrays, {} classes), walk {:.2}s",
        records,
        walk.instances,
        walk.prim_arrays,
        walk.object_arrays,
        walk.class_dumps,
        walk_elapsed.as_secs_f64()
    );
    let _ = segments;

    let mut heap = builder.build(id_size_byte)?;
    let started = std::time::Instant::now();
    match sink.finish(cfg) {
        Ok(dir) => {
            eprintln!("  Object directory ready in {:.2}s", started.elapsed().as_secs_f64());
            heap.obj_dir = Some(dir);
        }
        Err(e) => eprintln!(
            "Warning: object directory unavailable ({}); falling back to in-memory indexes",
            e
        ),
    }
    Ok(heap)
}

// ---------------------------------------------------------------------------
// File header
// ---------------------------------------------------------------------------

struct Header {
    /// Bytes of the header, reused as the prefix of every small-record batch.
    bytes: Vec<u8>,
    len: usize,
    id_size: u8,
    timestamp_millis: u64,
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        match file.read_at(&mut buf[done..], offset + done as u64) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short read while reading the HPROF header",
                ))
            }
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn read_some_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut done = 0usize;
    while done < buf.len() {
        match file.read_at(&mut buf[done..], offset + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

fn read_header(file: &File, file_len: u64) -> Result<Header> {
    let probe_len = file_len.min(256) as usize;
    if probe_len < MAGIC.len() + 1 {
        bail!("not a valid HPROF heap dump: file is too short ({file_len} bytes)");
    }
    let mut probe = vec![0u8; probe_len];
    read_exact_at(file, &mut probe, 0).context("cannot read the HPROF header")?;
    if !probe.starts_with(MAGIC) {
        bail!("not a valid HPROF heap dump: missing the \"JAVA PROFILE\" header");
    }
    let version = probe[MAGIC.len()];
    if version != b'2' && version != b'3' {
        bail!(
            "unsupported HPROF version 1.0.{}: this parser handles 1.0.2 and 1.0.3",
            version as char
        );
    }
    let nul = probe[MAGIC.len()..]
        .iter()
        .position(|&b| b == 0)
        .map(|p| p + MAGIC.len())
        .ok_or_else(|| anyhow!("truncated HPROF header: no version terminator"))?;
    let body_start = nul + 1;
    if body_start + 12 > probe_len {
        bail!("truncated HPROF header: no id size and timestamp");
    }
    let id_size = u32::from_be_bytes([
        probe[body_start],
        probe[body_start + 1],
        probe[body_start + 2],
        probe[body_start + 3],
    ]);
    if id_size != 4 && id_size != 8 {
        bail!("HPROF header id_size {id_size} is not 4 or 8");
    }
    let mut ts = [0u8; 8];
    ts.copy_from_slice(&probe[body_start + 4..body_start + 12]);
    let len = body_start + 12;
    Ok(Header {
        bytes: probe[..len].to_vec(),
        len,
        id_size: id_size as u8,
        timestamp_millis: u64::from_be_bytes(ts),
    })
}

// ---------------------------------------------------------------------------
// Forward scan
// ---------------------------------------------------------------------------

/// Forward reader that serves record headers and small record bodies from one
/// buffer, and never touches more of the file than the record it is on.
struct ScanReader {
    file: File,
    len: u64,
    buf: Vec<u8>,
    base: u64,
    filled: usize,
}

impl ScanReader {
    fn new(file: File, len: u64) -> Self {
        Self { file, len, buf: vec![0u8; SCAN_BUFFER], base: 0, filled: 0 }
    }

    fn file(&self) -> &File {
        &self.file
    }

    fn covers(&self, off: u64, n: u64) -> bool {
        off >= self.base && off + n <= self.base + self.filled as u64
    }

    fn refill(&mut self, off: u64) -> io::Result<()> {
        let want = (self.len - off).min(self.buf.len() as u64) as usize;
        let got = read_some_at(&self.file, &mut self.buf[..want], off)?;
        self.base = off;
        self.filled = got;
        Ok(())
    }

    /// Nine-byte top-level record header at off.
    ///
    /// The nine bytes come straight from the file when the forward buffer does not
    /// cover them, which is what keeps the scan cheap: a record of tens of
    /// megabytes costs one small read, not a buffer refill.
    fn header(&mut self, off: u64) -> io::Result<[u8; 9]> {
        let mut head = [0u8; 9];
        if self.covers(off, 9) {
            let s = (off - self.base) as usize;
            head.copy_from_slice(&self.buf[s..s + 9]);
        } else {
            read_exact_at(&self.file, &mut head, off)?;
        }
        Ok(head)
    }

    /// Append n bytes at off to out.
    fn append(&mut self, off: u64, mut n: u64, out: &mut Vec<u8>) -> io::Result<()> {
        let mut off = off;
        while n > 0 {
            if !self.covers(off, 1) {
                self.refill(off)?;
                if self.filled == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "truncated HPROF record body",
                    ));
                }
            }
            let s = (off - self.base) as usize;
            let avail = (self.filled - s) as u64;
            let take = avail.min(n) as usize;
            out.extend_from_slice(&self.buf[s..s + take]);
            off += take as u64;
            n -= take as u64;
        }
        Ok(())
    }
}

/// Decode the metadata records gathered since the last flush.
fn flush_small(
    small: &mut Vec<u8>,
    prefix: &[u8],
    id_size: IdSize,
    builder: &mut HeapDumpBuilder,
) -> Result<()> {
    let _ = id_size;
    if small.len() == prefix.len() {
        return Ok(());
    }
    let hprof = jvm_hprof::parse_hprof(small)
        .map_err(|e| anyhow!("Failed to parse HPROF: {:?}", e))?;
    for record in hprof.records_iter() {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: failed to parse record: {:?}", e);
                continue;
            }
        };
        builder.take_record(&record);
    }
    small.truncate(prefix.len());
    Ok(())
}

/// Walk one heap-dump record body, releasing its pages as the cursor passes them.
#[allow(clippy::too_many_arguments)]
fn walk_segment(
    file: &File,
    tag: u8,
    rec_offset: u64,
    body_off: u64,
    body_len: u64,
    id_size: u8,
    builder: &mut HeapDumpBuilder,
    sink: &mut ObjDirSink,
) -> Result<record_reader::WalkStats> {
    if body_len == 0 {
        return Ok(record_reader::WalkStats::default());
    }
    let len = usize::try_from(body_len)
        .map_err(|_| anyhow!("heap-dump record at offset {rec_offset} is too large to map"))?;
    // SAFETY: the mapping is only ever read through a shared slice.
    let map = unsafe { MmapOptions::new().offset(body_off).len(len).map(file) }
        .with_context(|| format!("cannot map the heap-dump record at offset {rec_offset}"))?;
    let data = &map[..];
    let origin = RegionOrigin::at(tag, rec_offset, body_off);
    let mut released = 0usize;
    let stats = record_reader::walk_heap_region(data, id_size, 0, data.len(), origin, &mut |item| {
        let local = match &item {
            HeapItem::Object(rec) => (rec.body_offset - body_off) as usize,
            HeapItem::Class(cls) => (cls.rec.aux_offset - body_off) as usize,
            HeapItem::Root(_) => released,
        };
        if local > released + RELEASE_WINDOW {
            let upto = local - RELEASE_WINDOW;
            // SAFETY: the range lies strictly behind the walk cursor inside a
            // read-only file mapping; a released page just faults back in.
            unsafe {
                let _ = map.unchecked_advise_range(
                    UncheckedAdvice::DontNeed,
                    released,
                    upto - released,
                );
            }
            released = upto;
        }
        match item {
            HeapItem::Root(root) => builder.add_root(&root),
            HeapItem::Object(rec) => return builder.add_object(&rec, sink),
            HeapItem::Class(cls) => return builder.register_class(&cls, id_size, sink),
        }
        Ok(())
    })?;
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// How many of the lowest object addresses the oop-size overlap validation
/// inspects: it walks the sorted address list and stops at this count.
const OVERLAP_SAMPLE: usize = 5000;

struct HeapDumpBuilder {
    string_table: HashMap<u64, String>,
    /// Class object id -> class name, from LOAD_CLASS plus the UTF8 table.
    load_classes: HashMap<u64, String>,
    stack_frames: HashMap<u64, jvm_hprof::StackFrame>,
    /// Stack trace serial -> (thread serial, frame ids).
    stack_traces: HashMap<u32, (u32, Vec<u64>)>,
    /// Class serial number -> class name string id.
    load_class_serials: HashMap<u32, u64>,
    roots: Vec<HeapRootReference>,
    classes: Vec<JavaClassInfo>,
    class_by_address: HashMap<u64, u32>,
    /// Synthetic "<type>[]" class index per HPROF basic type code.
    prim_class_idx: [u32; 12],
    /// The lowest `OVERLAP_SAMPLE` object addresses; the overlap validation
    /// only ever needs that prefix of the sorted list.
    smallest_addresses: std::collections::BinaryHeap<u64>,
    gen_counts: [u64; 4],
    heap_info: Option<HeapInfo>,
    object_count: u64,
    prim_array_count: u64,
    obj_array_count: u64,
    per_class_inst_count: Vec<u64>,
    per_class_inst_bytes: Vec<i64>,
    timestamp_millis: u64,
}

impl HeapDumpBuilder {
    fn new() -> Self {
        Self {
            string_table: HashMap::new(),
            load_classes: HashMap::new(),
            stack_frames: HashMap::new(),
            stack_traces: HashMap::new(),
            load_class_serials: HashMap::new(),
            roots: Vec::new(),
            classes: Vec::new(),
            class_by_address: HashMap::new(),
            prim_class_idx: [u32::MAX; 12],
            smallest_addresses: std::collections::BinaryHeap::with_capacity(OVERLAP_SAMPLE),
            gen_counts: [0; 4],
            heap_info: None,
            object_count: 0,
            prim_array_count: 0,
            obj_array_count: 0,
            per_class_inst_count: Vec::new(),
            per_class_inst_bytes: Vec::new(),
            timestamp_millis: 0,
        }
    }

    fn ensure_per_class(&mut self, class_idx: u32) {
        let needed = class_idx as usize + 1;
        if self.per_class_inst_count.len() < needed {
            self.per_class_inst_count.resize(needed, 0);
            self.per_class_inst_bytes.resize(needed, 0);
        }
    }

    /// Keep the lowest `OVERLAP_SAMPLE` addresses in a bounded max-heap.
    fn note_address(&mut self, addr: u64) {
        if self.smallest_addresses.len() < OVERLAP_SAMPLE {
            self.smallest_addresses.push(addr);
        } else if let Some(&max) = self.smallest_addresses.peek() {
            if addr < max {
                self.smallest_addresses.pop();
                self.smallest_addresses.push(addr);
            }
        }
    }

    /// Fold one metadata record into the tables the heap walk reads.
    fn take_record(&mut self, record: &jvm_hprof::Record<'_>) {
        match record.tag() {
            RecordTag::Utf8 => {
                if let Some(Ok(utf8)) = record.as_utf_8() {
                    let name_id = utf8.name_id().id();
                    let text = utf8.text_as_str().unwrap_or("???").to_string();
                    if name_id == 0 {
                        if let Some(info) = HeapInfo::parse(&text) {
                            self.heap_info = Some(info);
                        }
                    }
                    self.string_table.insert(name_id, text);
                }
            }
            RecordTag::LoadClass => {
                if let Some(Ok(lc)) = record.as_load_class() {
                    let name = self
                        .string_table
                        .get(&lc.class_name_id().id())
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    self.load_class_serials
                        .insert(lc.class_serial().num(), lc.class_name_id().id());
                    self.load_classes.insert(lc.class_obj_id().id(), name);
                }
            }
            RecordTag::StackFrame => {
                if let Some(Ok(sf)) = record.as_stack_frame() {
                    self.stack_frames.insert(sf.id().id(), sf);
                }
            }
            RecordTag::StackTrace => {
                if let Some(Ok(st)) = record.as_stack_trace() {
                    let frames: Vec<u64> = st
                        .frame_ids()
                        .filter_map(|f| f.ok().map(|id| id.id()))
                        .collect();
                    self.stack_traces
                        .insert(st.stack_trace_serial().num(), (st.thread_serial().num(), frames));
                }
            }
            _ => {}
        }
    }

    fn add_root(&mut self, root: &RootRec) {
        let root_type = match root.tag {
            0x01 => RootType::JniGlobal,
            0x02 => RootType::JniLocal,
            0x03 => RootType::JavaStackFrame,
            0x04 => RootType::NativeStack,
            0x05 => RootType::SystemClass,
            0x06 => RootType::ThreadBlock,
            0x07 => RootType::BusyMonitor,
            0x08 => RootType::ThreadObj,
            0xFF => RootType::Unknown,
            _ => return,
        };
        // A thread object root without an object id belongs to a thread newly
        // attached through JNI and carries no heap object.
        if root.tag == 0x08 && root.obj_id == 0 {
            return;
        }
        let carries_thread = matches!(root.tag, 0x02 | 0x03 | 0x04 | 0x06 | 0x08);
        let carries_frame = matches!(root.tag, 0x02 | 0x03);
        self.roots.push(HeapRootReference {
            root_type,
            object_id: root.obj_id,
            thread_serial: if carries_thread { Some(root.thread_serial) } else { None },
            frame_index: if carries_frame && root.frame_index != u32::MAX {
                Some(root.frame_index)
            } else {
                None
            },
        });
    }

    fn add_object(&mut self, rec: &WalkRec, sink: &mut ObjDirSink) -> Result<()> {
        match rec.kind {
            KIND_INSTANCE => {
                let class_idx = self.resolve_class_index(rec.class_ref);
                let generation = self.get_generation(rec.addr);
                self.gen_counts[generation as usize] += 1;
                self.note_address(rec.addr);
                self.object_count += 1;
                self.ensure_per_class(class_idx);
                self.per_class_inst_count[class_idx as usize] += 1;
                let inst_size = self
                    .classes
                    .get(class_idx as usize)
                    .map(|c| c.instance_size)
                    .unwrap_or(0);
                self.per_class_inst_bytes[class_idx as usize] += object_shallow_size(inst_size);
                sink.push(rec.addr, rec.body_offset, class_idx, 0, KIND_INSTANCE, 0, 0)?;
            }
            KIND_OBJ_ARRAY => {
                let class_idx = self.resolve_class_index(rec.class_ref);
                let generation = self.get_generation(rec.addr);
                self.gen_counts[generation as usize] += 1;
                self.note_address(rec.addr);
                self.obj_array_count += 1;
                sink.push(
                    rec.addr,
                    rec.body_offset,
                    class_idx,
                    rec.len,
                    KIND_OBJ_ARRAY,
                    rec.elem_size,
                    0,
                )?;
            }
            KIND_PRIM_ARRAY => {
                let class_idx = self.prim_class_index(rec.prim_type);
                let generation = self.get_generation(rec.addr);
                self.gen_counts[generation as usize] += 1;
                self.note_address(rec.addr);
                self.prim_array_count += 1;
                sink.push(
                    rec.addr,
                    rec.body_offset,
                    class_idx,
                    rec.len,
                    KIND_PRIM_ARRAY,
                    rec.elem_size,
                    rec.prim_type as u16,
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    fn register_class(
        &mut self,
        cls: &ClassRec<'_>,
        id_size: u8,
        sink: &mut ObjDirSink,
    ) -> Result<()> {
        let obj_id = cls.rec.addr;
        let name = self
            .load_classes
            .get(&obj_id)
            .cloned()
            .unwrap_or_else(|| "???".to_string());

        let mut instance_fields = Vec::with_capacity(cls.instance_count as usize);
        let mut fields = Fields::new(cls.instance_fields, id_size);
        while let Some((name_id, type_code)) = fields.next_instance() {
            let field_name = self
                .string_table
                .get(&name_id)
                .cloned()
                .unwrap_or_else(|| "???".to_string());
            instance_fields.push(FieldDesc { name: Arc::from(field_name), type_code });
        }

        let mut static_fields = Vec::with_capacity(cls.static_count as usize);
        let mut fields = Fields::new(cls.static_fields, id_size);
        while let Some((name_id, type_code)) = fields.next_static() {
            let field_name = self
                .string_table
                .get(&name_id)
                .cloned()
                .unwrap_or_else(|| "???".to_string());
            static_fields.push(FieldDesc { name: Arc::from(field_name), type_code });
        }

        let category = classify_class(&name);
        let class_idx = self.classes.len() as u32;
        self.classes.push(JavaClassInfo {
            name: Arc::from(name),
            class_object_id: obj_id,
            super_class_id: cls.super_id,
            instance_size: cls.instance_size as i32,
            instance_field_count: instance_fields.len() as i32,
            instance_fields,
            static_fields,
            category,
            index: class_idx,
        });
        self.class_by_address.insert(obj_id, class_idx);
        sink.push(obj_id, cls.rec.aux_offset, class_idx, 0, KIND_CLASS_DUMP, 0, 0)
    }

    fn resolve_class_index(&mut self, class_obj_id: u64) -> u32 {
        if let Some(&idx) = self.class_by_address.get(&class_obj_id) {
            return idx;
        }
        // Create a placeholder
        let idx = self.classes.len() as u32;
        let name = format!("unresolved_0x{:x}", class_obj_id);
        self.classes.push(JavaClassInfo {
            name: Arc::from(name),
            class_object_id: class_obj_id,
            super_class_id: 0,
            instance_size: 0,
            instance_field_count: 0,
            instance_fields: Vec::new(),
            static_fields: Vec::new(),
            category: ClassCategory::Other,
            index: idx,
        });
        self.class_by_address.insert(class_obj_id, idx);
        idx
    }

    /// Class index of the synthetic "<type>[]" class for one HPROF basic type.
    ///
    /// The lookup that creates the class on first use runs once per type instead of
    /// once per primitive array.
    fn prim_class_index(&mut self, type_code: u8) -> u32 {
        let slot = type_code as usize;
        if slot < self.prim_class_idx.len() && self.prim_class_idx[slot] != u32::MAX {
            return self.prim_class_idx[slot];
        }
        let name = match PRIM_ARRAY_NAMES.get(slot) {
            Some(n) if !n.is_empty() => *n,
            _ => return u32::MAX,
        };
        let idx = self.resolve_class_by_name(name);
        if slot < self.prim_class_idx.len() {
            self.prim_class_idx[slot] = idx;
        }
        idx
    }

    fn resolve_class_by_name(&mut self, name: &str) -> u32 {
        // Look for existing class by name
        for (i, c) in self.classes.iter().enumerate() {
            if c.name.as_ref() == name {
                return i as u32;
            }
        }
        // Create synthetic class
        let idx = self.classes.len() as u32;
        self.classes.push(JavaClassInfo {
            name: Arc::from(name.to_string()),
            class_object_id: 0,
            super_class_id: 0,
            instance_size: 0,
            instance_field_count: 0,
            instance_fields: Vec::new(),
            static_fields: Vec::new(),
            category: ClassCategory::Array,
            index: idx,
        });
        idx
    }

    fn get_generation(&self, _address: u64) -> u8 {
        self.heap_info
            .as_ref()
            .map(|h| h.get_generation(_address))
            .unwrap_or(0)
    }

    fn build(mut self, identifier_size: u8) -> Result<HeapDump> {
        let total_objects = self.object_count + self.prim_array_count + self.obj_array_count;

        eprintln!(
            "Parsed: {} classes, {} objects, {} prim arrays, {} obj arrays, {} roots",
            self.classes.len(),
            self.object_count,
            self.prim_array_count,
            self.obj_array_count,
            self.roots.len()
        );

        // Build class_by_name map
        let mut class_by_name = std::collections::HashMap::new();
        for cls in &self.classes {
            class_by_name.insert(cls.name.clone(), cls.index);
        }

        // Resolve stack traces: resolve frame string IDs to actual strings
        let mut stack_traces = std::collections::HashMap::new();
        for (_serial, (thread_serial, frame_ids)) in &self.stack_traces {
            let mut frames = Vec::new();
            for id in frame_ids {
                if let Some(sf) = self.stack_frames.get(id) {
                    let method_name = self
                        .string_table
                        .get(&sf.method_name_id().id())
                        .cloned()
                        .unwrap_or_else(|| "<unknown>".to_string());
                    let method_sig = self
                        .string_table
                        .get(&sf.method_signature_id().id())
                        .cloned()
                        .unwrap_or_default();
                    let source_file = self
                        .string_table
                        .get(&sf.source_file_name_id().id())
                        .cloned()
                        .unwrap_or_default();
                    let line_number = match sf.line_num() {
                        jvm_hprof::LineNum::Normal(n) => n as i32,
                        jvm_hprof::LineNum::Unknown => -1,
                        jvm_hprof::LineNum::CompiledMethod => -2,
                        jvm_hprof::LineNum::NativeMethod => -3,
                    };
                    let class_serial_num = sf.class_serial().num();
                    let class_name = self
                        .load_class_serials
                        .get(&class_serial_num)
                        .and_then(|name_id| self.string_table.get(name_id))
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    frames.push(ss_heap_model::heap_dump::StackFrameInfo {
                        method_name,
                        method_signature: method_sig,
                        source_file,
                        class_serial: class_serial_num,
                        class_name,
                        frame_id: *id,
                        line_number,
                    });
                }
            }
            // Skip empty stack traces — Java's HprofStringCache.buildStackTraceMap()
            // only includes traces with at least one frame
            if frames.is_empty() {
                continue;
            }
            stack_traces.insert(
                *thread_serial,
                ss_heap_model::heap_dump::ThreadStackTrace { thread_serial: *thread_serial, frames },
            );
        }

        // Build thread_serial_map from ThreadObj roots
        let mut thread_serial_map = std::collections::HashMap::new();
        for root in &self.roots {
            if root.root_type == RootType::ThreadObj {
                if let Some(ts) = root.thread_serial {
                    thread_serial_map.insert(ts, root.object_id);
                }
            }
        }

        eprintln!(
            "  Stack traces: {}, Thread serials: {}",
            stack_traces.len(),
            thread_serial_map.len()
        );

        // The overlap validation is the only consumer of the address list, and it
        // stops after OVERLAP_SAMPLE entries, so the sorted sample is built here
        // and dropped with this function instead of living in the HeapDump.
        let overlap_sample = std::mem::take(&mut self.smallest_addresses).into_sorted_vec();

        // Determine oop_size (reference size) and header_size.
        //
        // CRITICAL: the HPROF "Heap Info" record's heapOopSize reports the
        // *native machine word size* (8 on 64-bit), NOT the compressed-oop
        // size. For HotSpot JVMs with a heap < 32GB, compressed oops are in
        // use and the effective reference size is 4. Java derives refSize by
        // validating object-address overlap (HprofDumpReader.validatePointer
        // Sizes), not by trusting heapOopSize.
        //
        // So regardless of whether heap info is present, on a 64-bit dump we
        // run the same overlap validation: try (8,16); if objects would
        // overlap, fall back to compressed (4,12).
        let (oop_size, header_size) = if identifier_size == 4 {
            // 32-bit JVM
            eprintln!("  32-bit JVM: oop_size=4, header_size=8");
            (4u8, 8u8)
        } else {
            let validate = |hdr_size: u8| -> bool {
                let mut prev_end: u64 = 0;
                let mut checked: u32 = 0;
                for &addr in &overlap_sample {
                    if checked > 0 && addr < prev_end {
                        return false; // overlap
                    }
                    let min_size = ((hdr_size as u64 + 8) + 7) & !7;
                    prev_end = addr + min_size;
                    checked += 1;
                    if checked >= OVERLAP_SAMPLE as u32 {
                        break;
                    }
                }
                true
            };

            if let Some(info) = &self.heap_info {
                eprintln!(
                    "  Heap info record: heapOopSize={} (native word size, not refSize)",
                    info.oop_size
                );
            }

            // Try: headerSize=16 (implies refSize=8, no compressed oops)
            if validate(16) {
                eprintln!("  64-bit JVM: oop_size=8, header_size=16 (no compressed oops)");
                (8u8, 16u8)
            } else {
                eprintln!("  64-bit JVM (compressed oops): oop_size=4, header_size=12");
                (4u8, 12u8)
            }
        };

        Ok(HeapDump {
            classes: self.classes,
            class_by_name,
            roots: self.roots,
            total_object_count: total_objects,
            string_table: self.string_table,
            identifier_size,
            oop_size,
            alignment: self.heap_info.as_ref().map(|h| h.object_alignment).unwrap_or(8),
            header_size,
            per_class_inst_count: self.per_class_inst_count,
            per_class_inst_bytes: self.per_class_inst_bytes,
            stack_traces,
            thread_serial_map,
            thread_info: Vec::new(),
            timestamp_millis: self.timestamp_millis,
            obj_dir: None,
        })
    }
}

fn classify_class(name: &str) -> ClassCategory {
    if name == "java/lang/String" || name == "java.lang.String" {
        ClassCategory::String
    } else if (name.starts_with('[') && (name.ends_with('B') || name.ends_with('C') || name == "[B" || name == "[C"))
        || name == "byte[]" || name == "char[]"
    {
        ClassCategory::StringValue
    } else if name == "java/lang/Class" || name == "java.lang.Class" {
        ClassCategory::Class
    } else if name.contains("Thread") && !name.contains("ThreadPool") {
        ClassCategory::Thread
    } else if name.contains("ClassLoader") {
        ClassCategory::ClassLoader
    } else if name.starts_with('[') || name.ends_with("[]") {
        ClassCategory::Array
    } else if name == "java/lang/Integer" || name == "java/lang/Long"
        || name == "java/lang/Double" || name == "java/lang/Float"
        || name == "java/lang/Short" || name == "java/lang/Byte"
        || name == "java/lang/Boolean" || name == "java/lang/Character"
    {
        ClassCategory::BoxedNumber
    } else if name == "sun/misc/Cleaner" || name == "sun.misc.Cleaner" {
        ClassCategory::SunMiscCleaner
    } else if name == "jdk/internal/ref/Cleaner" || name == "jdk.internal.ref.Cleaner" {
        ClassCategory::JdkInternalRefCleaner
    } else if name == "java/lang/ThreadGroup" || name == "java.lang.ThreadGroup"
        || name == "java/lang/ThreadGroup$"
    {
        ClassCategory::ThreadGroup
    } else if name == "java/nio/DirectByteBuffer" || name == "java/nio/DirectByteBufferR"
    {
        // Match Java's ClassCategory.DIRECT_BB exactly: only the two buffer
        // classes that hold native memory. Typed views (DirectLongBufferU,
        // DirectIntBufferU, DirectCharBufferU/S, DirectFloatBufferU,
        // DirectDoubleBufferU, DirectShortBufferU) are NOT counted — they
        // share the underlying buffer's native memory and are excluded by
        // JavaClassInfo.java.
        ClassCategory::DirectByteBuffer
    } else {
        ClassCategory::Other
    }
}

/// Calculate JVM object shallow size: header + instance fields, aligned to 8 bytes.
fn object_shallow_size(instance_size: i32) -> i64 {
    // JVM object header: 12 bytes (mark word 8 + klass pointer 4) for compressed oops,
    // or 16 bytes for uncompressed. Use 12 as default, align to 8.
    let header: i64 = 12;
    let raw = header + instance_size as i64;
    (raw + 7) & !7
}
