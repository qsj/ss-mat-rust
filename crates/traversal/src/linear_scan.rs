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

/// Linear scan coordinator: re-parses the HPROF to access field data
/// and runs 4 parallel scan passes over instances and arrays.
pub struct LinearScanCoordinator {
    heap: Arc<HeapDump>,
    stats: Arc<StatsEngine>,
}

impl LinearScanCoordinator {
    pub fn new(heap: Arc<HeapDump>, stats: Arc<StatsEngine>) -> Self {
        Self { heap, stats }
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
        let mut prim_arrays: Vec<PrimArrayData> = Vec::new();
        let mut obj_arrays: Vec<ObjArrayData> = Vec::new();
        let mut direct_buffer_instances: Vec<DirectBufferData> = Vec::new();
        let mut bad_array_records: Vec<BadArrayRecord> = Vec::new();
        let mut class_obj_ids: HashMap<u64, u32> = HashMap::new();

        // Build primitive array body map for String content extraction: obj_id -> (body, elem_type)
        let mut prim_array_bodies: HashMap<u64, (&[u8], PrimitiveArrayType)> = HashMap::new();

        for cls in &self.heap.classes {
            if cls.class_object_id != 0 {
                class_obj_ids.insert(cls.class_object_id, cls.index);
            }
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
                                            if ref_field_nonnull(&fields, cls, "fd", &class_obj_ids, &self.heap.classes)
                                                || ref_field_nonnull(&fields, cls, "att", &class_obj_ids, &self.heap.classes)
                                            {
                                                continue;
                                            }
                                            if let Some(pc) = self.stats.get_per_class(class_idx) {
                                                let header = self.heap.header_size as i64;
                                                let shallow = (header + cls.instance_size as i64 + 7) & !7;
                                                pc.total_instance_bytes.fetch_add(shallow, Relaxed);
                                            }
                                            // Store for later aggregation
                                            direct_buffer_instances.push(DirectBufferData {
                                                class_idx,
                                                obj_id: inst.obj_id().id(),
                                                capacity,
                                                address,
                                            });
                                        } else if cat == ClassCategory::BoxedNumber {
                                            // Accumulate boxed number overhead
                                            if let Some(pc) = self.stats.get_per_class(class_idx) {
                                                let cls = &self.heap.classes[class_idx as usize];
                                                let header = self.heap.header_size as i64;
                                                let shallow = (header + cls.instance_size as i64 + 7) & !7;
                                                let prim_size = boxed_primitive_size(&cls.name) as i64;
                                                let overhead = shallow - prim_size;
                                                pc.boxed_number_overhead.fetch_add(overhead, Relaxed);
                                                pc.total_instance_bytes.fetch_add(shallow, Relaxed);
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
                                    prim_arrays.push(PrimArrayData {
                                        class_idx,
                                        obj_id,
                                        length,
                                        element_size,
                                        raw_body: raw_body.clone(),
                                        prim_type,
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

        // String scan — extract String content for dedup analysis
        let string_result: Vec<(u64, StringHashEntry)> = string_instances
            .par_iter()
            .enumerate()
            .filter_map(|(i, inst)| {
                if inst.class_idx as usize >= self.heap.classes.len() {
                    return None;
                }
                let content = extract_string_content_with_body(
                    &inst.fields,
                    &self.heap.classes[inst.class_idx as usize],
                    id_size,
                    &prim_body_map,
                )?;
                let hash = compute_string_hash(&content);
                Some((inst.obj_id, StringHashEntry { content, hash, index: i }))
            })
            .collect();

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
                    // Track bad primitive arrays
                    if arr.length == 0 {
                        stats_ref.empty_prim_array_count.fetch_add(1, Relaxed);
                        stats_ref.empty_prim_array_bytes.fetch_add(size, Relaxed);
                        push_sample(&stats_ref.empty_prim_array_samples, arr.obj_id);
                    } else if arr.length > 1 && has_trailing_zero_elements(&arr.raw_body, arr.element_size) {
                        stats_ref.trailing_zero_prim_array_count.fetch_add(1, Relaxed);
                        stats_ref.trailing_zero_prim_array_bytes.fetch_add(size, Relaxed);
                        push_sample(&stats_ref.trailing_zero_prim_array_samples, arr.obj_id);
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
                    // Track bad object arrays (matching Java's 5 categories).
                    // Capture a record so the report can later filter to
                    // live-only using BFS `visited` (Java only counts live).
                    if arr.length == 0 {
                        bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 0, size });
                    } else if arr.length == 1 {
                        bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 1, size });
                    } else {
                        let non_null_count = arr.element_refs.len() as i32;
                        let null_count = arr.length - non_null_count;
                        if non_null_count == 0 {
                            bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 2, size });
                        } else if non_null_count == 1 {
                            bad_array_records.push(BadArrayRecord { obj_id: arr.obj_id, class_idx: arr.class_idx, kind: 3, size });
                        } else if null_count as f64 > 0.70 * arr.length as f64 {
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

        // Report string scan results
        let dup_strings = find_duplicate_strings(&string_result);
        eprintln!(
            "String scan: {} total, {} duplicate groups",
            string_result.len(),
            dup_strings.len()
        );

        // Build groups for the StringDupCollector
        let total_string_count = string_result.len();
        let unique_count = total_string_count - dup_strings.iter().map(|(_, c, _)| c - 1).sum::<usize>();

        // Java's duplicate-string waste formula (StringContentDeduplicator):
        //   groupWaste = (count-1)*stringBodySize
        //              + (count-1)*backingArraySize
        // where stringBodySize = align8(String instance fields + header), and
        // backingArraySize = align8(objectAlignBase + elementSize*contentLen).
        // elementSize is 1 for byte[] (Java 9+ compact strings, Latin-1).
        let str_idx = self.heap.class_by_name.get("java/lang/String").copied();
        let header = self.heap.header_size as i64;
        let object_align_base = header + 4; // header + array length field
        let string_body_size: i64 = str_idx
            .and_then(|idx| self.heap.classes.get(idx as usize))
            .map(|cls| {
                let raw = header + cls.instance_size as i64;
                (raw + 7) & !7 // align8
            })
            .unwrap_or(24); // fallback: hash(4)+coder(1)+value(4)+header(12)=24

        let groups: Vec<ss_stats::duplicate_strings::DupStringGroup> = dup_strings
            .iter()
            .map(|(content, count, samples)| {
                let cnt = *count as i64;
                let extra = cnt - 1; // the duplicate copies beyond the first
                // Term A: extra String object bodies.
                let waste_strings = extra * string_body_size;
                // Term B: extra backing byte[] arrays. content.len() chars;
                // Java 9+ compact strings store as byte[] (1 byte/char for Latin-1).
                let array_raw = object_align_base + content.len() as i64;
                let array_size = (array_raw + 7) & !7; // align8
                let waste_arrays = extra * array_size;
                let wasted = waste_strings + waste_arrays;
                ss_stats::duplicate_strings::DupStringGroup {
                    content: content.clone(),
                    count: *count,
                    wasted_bytes: wasted,
                    sample_obj_ids: samples.clone(),
                }
            })
            .collect();

        // Accumulate duplicate string bytes
        for g in &groups {
            if let Some(&str_idx) = self.heap.class_by_name.get("java/lang/String") {
                if let Some(pc) = stats_ref.get_per_class(str_idx) {
                    pc.duplicate_string_bytes.fetch_add(g.wasted_bytes, Relaxed);
                }
            }
        }

        // Store results in StatsEngine
        self.stats.string_dup.set_results(total_string_count, unique_count, groups);

        // DirectByteBuffer off-heap analysis
        aggregate_direct_buffers(&direct_buffer_instances, &self.heap.classes, &self.stats);
        // Stash all bad-array records (live+garbage) for later live-only filtering.
        *self.stats.bad_array_records.lock().unwrap() = bad_array_records;

        // Duplicate primitive arrays detection
        detect_duplicate_prim_arrays(&prim_arrays, &self.heap.classes, &self.stats, self.heap.header_size as i64);

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

struct PrimArrayData {
    class_idx: u32,
    obj_id: u64,
    length: i32,
    element_size: i32,
    raw_body: Vec<u8>,
    prim_type: PrimitiveArrayType,
}

struct ObjArrayData {
    class_idx: u32,
    obj_id: u64,
    length: i32,
    element_refs: Vec<u64>,
}

struct StringHashEntry {
    content: String,
    hash: u64,
    index: usize,
}

/// Extract String content from instance field bytes using primitive array body data.
fn extract_string_content_with_body(
    fields: &[u8],
    class_info: &JavaClassInfo,
    id_size: IdSize,
    prim_body_map: &HashMap<u64, (Vec<u8>, PrimitiveArrayType)>,
) -> Option<String> {
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

    // Look up the backing array body
    let (body, prim_type) = prim_body_map.get(&value_id)?;

    // Extract string content based on coder / array type
    // Java 9+: coder=0 -> LATIN1 (byte[]), coder=1 -> UTF16 (byte[])
    // Java 8: uses char[]
    if *prim_type == PrimitiveArrayType::Char {
        // Java 8 style: char[] backing
        let chars: Vec<u16> = body.chunks(2)
            .filter_map(|c| Some(u16::from_be_bytes(c.try_into().ok()?)))
            .collect();
        let start = str_offset.max(0) as usize;
        let end = (str_offset + str_count.min(chars.len() as i32)).min(chars.len() as i32) as usize;
        if start >= end || start >= chars.len() {
            return None;
        }
        chars[start..end].iter().map(|&c| char::from_u32(c as u32)).collect()
    } else if *prim_type == PrimitiveArrayType::Byte {
        if coder == 0 {
            // LATIN1 encoding (Java 9+)
            let bytes = &body[..];
            let start = str_offset.max(0) as usize;
            let end = (str_count.min(bytes.len() as i32)) as usize;
            if start >= end || start >= bytes.len() {
                return None;
            }
            Some(bytes[start..end].iter().map(|&b| b as char).collect())
        } else {
            // UTF16 encoding (Java 9+)
            let chars: Vec<u16> = body.chunks(2)
                .filter_map(|c| Some(u16::from_be_bytes(c.try_into().ok()?)))
                .collect();
            let start = str_offset.max(0) as usize;
            let end = (str_count.min(chars.len() as i32)) as usize;
            if start >= end || start >= chars.len() {
                return None;
            }
            chars[start..end].iter().map(|&c| char::from_u32(c as u32)).collect()
        }
    } else {
        None
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

fn compute_string_hash(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn find_duplicate_strings(entries: &[(u64, StringHashEntry)]) -> Vec<(String, usize, Vec<u64>)> {
    let mut groups: HashMap<u64, Vec<&(u64, StringHashEntry)>> = HashMap::new();
    for pair in entries {
        groups.entry(pair.1.hash).or_default().push(pair);
    }

    let mut dups = Vec::new();
    for (_, group) in groups {
        if group.len() > 1 {
            // Collect up to 5 sample obj IDs per group (for ref chain rendering).
            let samples: Vec<u64> = group.iter().take(5).map(|(obj_id, _)| *obj_id).collect();
            dups.push((group[0].1.content.clone(), group.len(), samples));
        }
    }
    dups.sort_by(|a, b| b.1.cmp(&a.1));
    dups
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

fn detect_duplicate_prim_arrays(
    prim_arrays: &[PrimArrayData],
    classes: &[JavaClassInfo],
    stats: &StatsEngine,
    header_size: i64,
) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Group by (class_idx, body_hash) -> count
    let mut groups: HashMap<(u32, u64), Vec<&PrimArrayData>> = HashMap::new();
    for arr in prim_arrays {
        if arr.raw_body.len() > 32 {
            // Only check arrays > 32 bytes (skip tiny)
            let mut hasher = DefaultHasher::new();
            arr.raw_body.hash(&mut hasher);
            let hash = hasher.finish();
            groups.entry((arr.class_idx, hash)).or_default().push(arr);
        }
    }

    // Verify actual body equality and build per-class stats
    let mut per_class: HashMap<u32, (usize, usize, i64)> = HashMap::new(); // class_idx -> (total, unique, wasted)
    let mut top_groups: Vec<ss_stats::duplicate_arrays::DupArrayGroup> = Vec::new();
    let mut total_count: usize = 0;

    for ((class_idx, _), arrs) in &groups {
        if arrs.len() < 2 { continue; }

        // Verify actual body equality within the hash bucket
        let mut confirmed_groups: Vec<Vec<&PrimArrayData>> = Vec::new();
        for arr in arrs {
            let mut placed = false;
            for group in confirmed_groups.iter_mut() {
                if group[0].raw_body == arr.raw_body {
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
            let size = array_shallow_size(group[0].length, group[0].element_size, header_size);
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
                    description: format!("byte[{}]", group[0].length),
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
        "Duplicate prim arrays: {} total in {} groups",
        total_count,
        per_class_entries.len()
    );

    stats.dup_prim_arrays.set_results(total_count, unique_total, per_class_entries, top_groups);
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

/// Check if a primitive array has trailing zero elements (>50% of elements are zero at the tail).
fn has_trailing_zero_elements(raw_body: &[u8], element_size: i32) -> bool {
    if raw_body.is_empty() || element_size <= 0 {
        return false;
    }
    let esz = element_size as usize;
    let total_elements = raw_body.len() / esz;
    if total_elements < 2 {
        return false;
    }
    // Count trailing zero elements
    let mut trailing = 0usize;
    for i in (0..total_elements).rev() {
        let start = i * esz;
        let end = start + esz;
        if end <= raw_body.len() && raw_body[start..end].iter().all(|&b| b == 0) {
            trailing += 1;
        } else {
            break;
        }
    }
    trailing * 2 > total_elements // more than 50% trailing zeros
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
