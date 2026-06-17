use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use hashbrown::HashMap as FxHashMap;
use jvm_hprof::{heap_dump, Id, IdSize, RecordTag};
use memmap2::Mmap;

use ss_heap_model::class_info::{ClassCategory, FieldDesc, JavaClassInfo};
use ss_heap_model::heap_dump::HeapDump;
use ss_heap_model::root::{HeapRootReference, RootType};

use crate::heap_info::HeapInfo;
use crate::object_index::ObjectIndex;

/// Parse an HPROF file and produce a HeapDump.
pub fn parse_hprof(path: &Path) -> Result<HeapDump> {
    let file = std::fs::File::open(path).with_context(|| format!("Failed to open: {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data = &mmap[..];

    let hprof = jvm_hprof::parse_hprof(data)
        .map_err(|e| anyhow::anyhow!("Failed to parse HPROF: {:?}", e))?;

    let id_size = hprof.header().id_size();
    let id_size_byte: u8 = match id_size {
        IdSize::U32 => 4,
        IdSize::U64 => 8,
    };

    eprintln!("HPROF parsed: id_size={}", id_size_byte);

    // Single pass through all records to build lookup tables and index
    let mut builder = HeapDumpBuilder::new(id_size);
    builder.timestamp_millis = hprof.header().timestamp_millis();

    for record in hprof.records_iter() {
        let record = match record {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: failed to parse record: {:?}", e);
                continue;
            }
        };

        match record.tag() {
            RecordTag::Utf8 => {
                if let Some(Ok(utf8)) = record.as_utf_8() {
                    let name_id = utf8.name_id().id();
                    let text = utf8.text_as_str().unwrap_or("???").to_string();

                    // The special name_id=0 STRING record contains JVM heap info
                    if name_id == 0 {
                        if let Some(info) = HeapInfo::parse(&text) {
                            builder.heap_info = Some(info);
                        }
                    }

                    builder.string_table.insert(name_id, text);
                }
            }
            RecordTag::LoadClass => {
                if let Some(Ok(lc)) = record.as_load_class() {
                    let name = builder
                        .string_table
                        .get(&lc.class_name_id().id())
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    builder.load_class_serials.insert(lc.class_serial().num(), lc.class_name_id().id());
                    builder.load_classes.insert(lc.class_obj_id(), (lc, name));
                }
            }
            RecordTag::StackFrame => {
                if let Some(Ok(sf)) = record.as_stack_frame() {
                    builder.stack_frames.insert(sf.id(), sf);
                }
            }
            RecordTag::StackTrace => {
                if let Some(Ok(st)) = record.as_stack_trace() {
                    builder.stack_traces.insert(st.stack_trace_serial(), st);
                }
            }
            RecordTag::HeapDump | RecordTag::HeapDumpSegment => {
                if let Some(Ok(seg)) = record.as_heap_dump_segment() {
                    for sr in seg.sub_records() {
                        match sr {
                            Ok(heap_dump::SubRecord::GcRootJniGlobal(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::JniGlobal,
                                    object_id: r.obj_id().id(),
                                    thread_serial: None,
                                    frame_index: None,
                                });
                            }
                            Ok(heap_dump::SubRecord::GcRootJniLocalRef(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::JniLocal,
                                    object_id: r.obj_id().id(),
                                    thread_serial: Some(r.thread_serial().num()),
                                    frame_index: r.frame_index(),
                                });
                            }
                            Ok(heap_dump::SubRecord::GcRootJavaStackFrame(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::JavaStackFrame,
                                    object_id: r.obj_id().id(),
                                    thread_serial: Some(r.thread_serial().num()),
                                    frame_index: r.frame_index(),
                                });
                            }
                            Ok(heap_dump::SubRecord::GcRootNativeStack(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::NativeStack,
                                    object_id: r.obj_id().id(),
                                    thread_serial: Some(r.thread_serial().num()),
                                    frame_index: None,
                                });
                            }
                            Ok(heap_dump::SubRecord::GcRootSystemClass(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::SystemClass,
                                    object_id: r.obj_id().id(),
                                    thread_serial: None,
                                    frame_index: None,
                                });
                            }
                            Ok(heap_dump::SubRecord::GcRootThreadBlock(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::ThreadBlock,
                                    object_id: r.obj_id().id(),
                                    thread_serial: Some(r.thread_serial().num()),
                                    frame_index: None,
                                });
                            }
                            Ok(heap_dump::SubRecord::GcRootBusyMonitor(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::BusyMonitor,
                                    object_id: r.obj_id().id(),
                                    thread_serial: None,
                                    frame_index: None,
                                });
                            }
                            Ok(heap_dump::SubRecord::GcRootThreadObj(r)) => {
                                if let Some(obj_id) = r.thread_obj_id() {
                                    builder.roots.push(HeapRootReference {
                                        root_type: RootType::ThreadObj,
                                        object_id: obj_id.id(),
                                        thread_serial: Some(r.thread_serial().num()),
                                        frame_index: None,
                                    });
                                }
                            }
                            Ok(heap_dump::SubRecord::GcRootUnknown(r)) => {
                                builder.roots.push(HeapRootReference {
                                    root_type: RootType::Unknown,
                                    object_id: r.obj_id().id(),
                                    thread_serial: None,
                                    frame_index: None,
                                });
                            }
                            Ok(heap_dump::SubRecord::Class(class_dump)) => {
                                builder.register_class(class_dump, id_size);
                            }
                            Ok(heap_dump::SubRecord::Instance(inst)) => {
                                builder.add_instance(inst);
                            }
                            Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                                builder.add_object_array(arr, id_size);
                            }
                            Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                                builder.add_primitive_array(arr);
                            }
                            Err(e) => {
                                eprintln!("Warning: failed to parse sub-record: {:?}", e);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    builder.build(id_size_byte)
}

struct HeapDumpBuilder<'a> {
    string_table: HashMap<u64, String>,
    load_classes: HashMap<Id, (jvm_hprof::LoadClass, String)>,
    stack_frames: HashMap<Id, jvm_hprof::StackFrame>,
    stack_traces: HashMap<jvm_hprof::Serial, jvm_hprof::StackTrace<'a>>,
    load_class_serials: HashMap<u32, u64>, // class_serial_number -> class_name_string_id
    roots: Vec<HeapRootReference>,
    classes: Vec<JavaClassInfo>,
    class_by_address: HashMap<u64, u32>,
    object_index_builder: ObjectIndexBuilder,
    address_to_index: FxHashMap<u64, u32, ahash::RandomState>,
    gen_counts: [u64; 4],
    heap_info: Option<HeapInfo>,
    object_count: u64,
    prim_array_count: u64,
    obj_array_count: u64,
    // Per-class instance counts and total shallow bytes (populated during parse)
    per_class_inst_count: Vec<u64>,
    per_class_inst_bytes: Vec<i64>,
    // Object address -> class_idx for local variable resolution
    object_to_class_idx: HashMap<u64, u32>,
    // HPROF timestamp
    timestamp_millis: u64,
}

impl<'a> HeapDumpBuilder<'a> {
    fn new(id_size: IdSize) -> Self {
        let _ = id_size;
        Self {
            string_table: HashMap::new(),
            load_classes: HashMap::new(),
            stack_frames: HashMap::new(),
            stack_traces: HashMap::new(),
            load_class_serials: HashMap::new(),
            roots: Vec::new(),
            classes: Vec::new(),
            class_by_address: HashMap::new(),
            object_index_builder: ObjectIndexBuilder::new(),
            address_to_index: FxHashMap::with_hasher(ahash::RandomState::new()),
            gen_counts: [0; 4],
            heap_info: None,
            object_count: 0,
            prim_array_count: 0,
            obj_array_count: 0,
            per_class_inst_count: Vec::new(),
            per_class_inst_bytes: Vec::new(),
            object_to_class_idx: HashMap::new(),
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

    fn register_class(&mut self, class_dump: heap_dump::Class, id_size: IdSize) {
        let obj_id = class_dump.obj_id();
        let name = self
            .load_classes
            .get(&obj_id)
            .map(|(_, n)| n.as_str())
            .unwrap_or("???");

        let super_class_id = class_dump.super_class_obj_id().map(|id| id.id()).unwrap_or(0);

        let mut instance_fields = Vec::new();
        for fd in class_dump.instance_field_descriptors() {
            match fd {
                Ok(fd) => {
                    let field_name = self
                        .string_table
                        .get(&fd.name_id().id())
                        .cloned()
                        .unwrap_or_else(|| "???".to_string());
                    let type_code = match fd.field_type() {
                        heap_dump::FieldType::ObjectId => 2,
                        heap_dump::FieldType::Boolean => 4,
                        heap_dump::FieldType::Char => 5,
                        heap_dump::FieldType::Float => 6,
                        heap_dump::FieldType::Double => 7,
                        heap_dump::FieldType::Byte => 8,
                        heap_dump::FieldType::Short => 9,
                        heap_dump::FieldType::Int => 10,
                        heap_dump::FieldType::Long => 11,
                    };
                    instance_fields.push(FieldDesc {
                        name: Arc::from(field_name),
                        type_code,
                    });
                }
                Err(e) => {
                    eprintln!("Warning: failed to parse field descriptor: {:?}", e);
                }
            }
        }

        let mut static_fields = Vec::new();
        for sfe in class_dump.static_fields() {
            match sfe {
                Ok(sfe) => {
                    let field_name = self
                        .string_table
                        .get(&sfe.name_id().id())
                        .cloned()
                        .unwrap_or_else(|| "???".to_string());
                    let type_code = match sfe.field_type() {
                        heap_dump::FieldType::ObjectId => 2,
                        heap_dump::FieldType::Boolean => 4,
                        heap_dump::FieldType::Char => 5,
                        heap_dump::FieldType::Float => 6,
                        heap_dump::FieldType::Double => 7,
                        heap_dump::FieldType::Byte => 8,
                        heap_dump::FieldType::Short => 9,
                        heap_dump::FieldType::Int => 10,
                        heap_dump::FieldType::Long => 11,
                    };
                    static_fields.push(FieldDesc {
                        name: Arc::from(field_name),
                        type_code,
                    });
                }
                Err(e) => {
                    eprintln!("Warning: failed to parse static field: {:?}", e);
                }
            }
        }

        let category = classify_class(name);

        let class_idx = self.classes.len() as u32;
        let class_info = JavaClassInfo {
            name: Arc::from(name.to_string()),
            class_object_id: obj_id.id(),
            super_class_id,
            instance_size: class_dump.instance_size_bytes() as i32,
            instance_field_count: instance_fields.len() as i32,
            instance_fields,
            static_fields,
            category,
            index: class_idx,
        };

        self.classes.push(class_info);
        self.class_by_address.insert(obj_id.id(), class_idx);

        let _ = id_size; // used later for field value reading
    }

    fn add_instance(&mut self, inst: heap_dump::Instance) {
        let class_obj_id = inst.class_obj_id();
        let obj_id = inst.obj_id();
        let class_idx = self.resolve_class_index(class_obj_id.id());
        let generation = self.get_generation(obj_id.id());

        self.gen_counts[generation as usize] += 1;
        let index = self.object_index_builder.add_object(class_idx, 0, generation);
        self.address_to_index.insert(obj_id.id(), index);
        self.object_to_class_idx.insert(obj_id.id(), class_idx);
        self.object_count += 1;

        // Per-class accumulation
        self.ensure_per_class(class_idx);
        self.per_class_inst_count[class_idx as usize] += 1;
        let inst_size = self.classes.get(class_idx as usize).map(|c| c.instance_size).unwrap_or(0);
        let shallow = object_shallow_size(inst_size);
        self.per_class_inst_bytes[class_idx as usize] += shallow;
    }

    fn add_object_array(&mut self, arr: heap_dump::ObjectArray, id_size: IdSize) {
        let obj_id = arr.obj_id();
        let array_class_id = arr.array_class_obj_id();
        let class_idx = self.resolve_class_index(array_class_id.id());
        let generation = self.get_generation(obj_id.id());

        let length = arr.elements(id_size).count() as i32;

        self.gen_counts[generation as usize] += 1;
        let index = self.object_index_builder.add_array(class_idx, 0, length, generation);
        self.address_to_index.insert(obj_id.id(), index);
        self.object_to_class_idx.insert(obj_id.id(), class_idx);
        self.obj_array_count += 1;
    }

    fn add_primitive_array(&mut self, arr: heap_dump::PrimitiveArray) {
        let obj_id = arr.obj_id();
        let type_name = arr.primitive_type().java_type_name();
        let class_name = format!("{}[]", type_name);
        let class_idx = self.resolve_class_by_name(&class_name);
        let generation = self.get_generation(obj_id.id());

        let length = count_primitive_array_elements(&arr);

        self.gen_counts[generation as usize] += 1;
        let index = self.object_index_builder.add_array(class_idx, 0, length, generation);
        self.address_to_index.insert(obj_id.id(), index);
        self.object_to_class_idx.insert(obj_id.id(), class_idx);
        self.prim_array_count += 1;
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
        // Use heap_info if available, otherwise default to 0
        self.heap_info
            .as_ref()
            .map(|h| h.get_generation(_address))
            .unwrap_or(0)
    }

    fn build(self, identifier_size: u8) -> Result<HeapDump> {
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

        let _object_index = self.object_index_builder.build(self.classes.len() as u32);

        // Resolve stack traces: resolve frame string IDs to actual strings
        let mut stack_traces = std::collections::HashMap::new();
        for (_serial, st) in &self.stack_traces {
            let thread_serial = st.thread_serial().num();
            let mut frames = Vec::new();
            for fid in st.frame_ids() {
                if let Ok(id) = fid {
                    if let Some(sf) = self.stack_frames.get(&id) {
                        let method_name = self.string_table
                            .get(&sf.method_name_id().id())
                            .cloned()
                            .unwrap_or_else(|| "<unknown>".to_string());
                        let method_sig = self.string_table
                            .get(&sf.method_signature_id().id())
                            .cloned()
                            .unwrap_or_else(|| "".to_string());
                        let source_file = self.string_table
                            .get(&sf.source_file_name_id().id())
                            .cloned()
                            .unwrap_or_else(|| "".to_string());
                        let line_number = match sf.line_num() {
                            jvm_hprof::LineNum::Normal(n) => n as i32,
                            jvm_hprof::LineNum::Unknown => -1,
                            jvm_hprof::LineNum::CompiledMethod => -2,
                            jvm_hprof::LineNum::NativeMethod => -3,
                        };
                        let class_serial_num = sf.class_serial().num();
                        let class_name = self.load_class_serials
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
                            frame_id: id.id(),
                            line_number,
                        });
                    }
                }
            }
            // Skip empty stack traces — Java's HprofStringCache.buildStackTraceMap()
            // only includes traces with at least one frame
            if frames.is_empty() {
                continue;
            }
            stack_traces.insert(thread_serial, ss_heap_model::heap_dump::ThreadStackTrace {
                thread_serial,
                frames,
            });
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

        // Determine oop_size and header_size.
        // If heap info was parsed (heapOopSize found), use it directly.
        // Otherwise, validate by checking object address overlap with different size assumptions.
        // This matches Java's HprofDumpReader.buildHeapDump() logic.
        let (oop_size, header_size) = if let Some(info) = &self.heap_info {
            let oop = info.oop_size;
            let hdr = if oop == 8 { 16u8 } else { 12u8 };
            eprintln!("  Heap info: oop_size={}, header_size={}", oop, hdr);
            (oop, hdr)
        } else if identifier_size == 4 {
            // 32-bit JVM
            eprintln!("  32-bit JVM: oop_size=4, header_size=8");
            (4u8, 8u8)
        } else {
            // 64-bit JVM, no heap info — validate using object address overlap.
            // Collect sorted object addresses and check whether objects would overlap
            // with different (refSize, headerSize) assumptions.
            // Java's approach: try (8,16), if overlap → try (4,12).
            let mut addrs: Vec<u64> = self.address_to_index.keys().copied().collect();
            addrs.sort_unstable();

            // For each address, estimate minimum object size given headerSize.
            // A typical small object: header + 0..8 bytes instance data = 16 or 24 bytes.
            // The key check: does the next address fall within the current object?
            let validate = |hdr_size: u8| -> bool {
                let mut prev_end: u64 = 0;
                let mut checked: u32 = 0;
                for &addr in &addrs {
                    if checked > 0 && addr < prev_end {
                        return false; // overlap
                    }
                    // Minimum aligned size with this header
                    let min_size = ((hdr_size as u64 + 8) + 7) & !7;
                    prev_end = addr + min_size;
                    checked += 1;
                    if checked >= 5000 { break; }
                }
                true
            };

            // Try: headerSize=16 (implies refSize=8, no compressed oops)
            if validate(16) {
                eprintln!("  64-bit JVM: oop_size=8, header_size=16");
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
            address_to_index: self.address_to_index,
            stack_traces,
            thread_serial_map,
            thread_info: Vec::new(),
            object_to_class_idx: self.object_to_class_idx,
            timestamp_millis: self.timestamp_millis,
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

fn count_primitive_array_elements(arr: &heap_dump::PrimitiveArray) -> i32 {
    if arr.booleans().is_some() {
        arr.booleans().map(|it| it.count() as i32).unwrap_or(0)
    } else if arr.bytes().is_some() {
        arr.bytes().map(|it| it.count() as i32).unwrap_or(0)
    } else if arr.chars().is_some() {
        arr.chars().map(|it| it.count() as i32).unwrap_or(0)
    } else if arr.shorts().is_some() {
        arr.shorts().map(|it| it.count() as i32).unwrap_or(0)
    } else if arr.ints().is_some() {
        arr.ints().map(|it| it.count() as i32).unwrap_or(0)
    } else if arr.longs().is_some() {
        arr.longs().map(|it| it.count() as i32).unwrap_or(0)
    } else if arr.floats().is_some() {
        arr.floats().map(|it| it.count() as i32).unwrap_or(0)
    } else if arr.doubles().is_some() {
        arr.doubles().map(|it| it.count() as i32).unwrap_or(0)
    } else {
        0
    }
}

/// Simple object index builder that stores per-object metadata in flat vectors.
struct ObjectIndexBuilder {
    // Per-object metadata: (class_idx, generation)
    objects: Vec<(u32, u8)>,
    // Per-array metadata: (class_idx, generation, length)
    arrays: Vec<(u32, u8, i32)>,
}

impl ObjectIndexBuilder {
    fn new() -> Self {
        Self {
            objects: Vec::new(),
            arrays: Vec::new(),
        }
    }

    /// Add an object dump instance. Returns a 1-based index (bit 31 clear).
    fn add_object(&mut self, class_idx: u32, _file_offset: u64, generation: u8) -> u32 {
        let index = (self.objects.len() + 1) as u32;
        self.objects.push((class_idx, generation));
        index
    }

    /// Add an array. Returns a 1-based index with bit 31 set.
    fn add_array(&mut self, class_idx: u32, _file_offset: u64, length: i32, generation: u8) -> u32 {
        let index = (self.arrays.len() + 1) as u32;
        self.arrays.push((class_idx, generation, length));
        index | 0x80000000
    }

    fn build(self, _class_count: u32) -> ObjectIndex {
        ObjectIndex::new(self.objects, self.arrays)
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
