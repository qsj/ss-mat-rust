use std::collections::VecDeque;
use std::collections::hash_map::HashMap;
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use anyhow::Result;
use jvm_hprof::{heap_dump, IdSize, RecordTag};
use memmap2::Mmap;

use ss_heap_model::class_info::ClassCategory;
use ss_heap_model::heap_dump::HeapDump;
use ss_heap_model::root::{HeapRootReference, RootType};
use ss_stats::engine::StatsEngine;
use ss_stats::ref_chain::{ChainStep, ClassRefChains, RefChainGroup, RefChainStore};

/// BFS object graph traverser. Walks from GC roots, marks visited objects,
/// extracts outgoing references, and accumulates retained sizes per class.
pub struct HeapObjectTraverser {
    heap: Arc<HeapDump>,
    stats: Arc<StatsEngine>,
}

/// Per-object metadata collected during initial HPROF pass for BFS.
struct ObjectEntry {
    obj_id: u64,
    class_idx: u32,
    file_offset: usize,
    body_size: i32,
    /// Object type: 0=instance, 1=primitive array, 2=object array, 3=class object
    kind: u8,
    array_length: i32,
}

/// Field name pool: compact storage of field names used in edges.
/// Index 0 = "[elements]" (array elements), 1 = "[static]" (static field).
struct FieldPool {
    names: Vec<String>,
}

impl FieldPool {
    fn new() -> Self {
        Self {
            names: vec!["[elements]".into(), "[static]".into()],
        }
    }
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(idx) = self.names.iter().position(|n| n == name) {
            return idx as u32;
        }
        let idx = self.names.len() as u32;
        self.names.push(name.to_string());
        idx
    }
    fn get(&self, key: u32) -> &str {
        &self.names[key as usize]
    }
}

/// Edge in the object graph carrying field info.
#[derive(Clone)]
struct OutEdge {
    target: u32,
    field_key: u32, // index into FieldPool
}

impl HeapObjectTraverser {
    pub fn new(heap: Arc<HeapDump>, stats: Arc<StatsEngine>) -> Self {
        Self { heap, stats }
    }

    /// Build a compact object table by re-parsing the HPROF once, then run BFS.
    pub fn run_traversal(&self, hprof_path: &Path) -> Result<TraversalResult> {
        let file = std::fs::File::open(hprof_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data = &mmap[..];

        let hprof = jvm_hprof::parse_hprof(data)
            .map_err(|e| anyhow::anyhow!("Failed to re-parse HPROF for BFS: {:?}", e))?;

        let id_size = hprof.header().id_size();
        let id_bytes: usize = match id_size {
            IdSize::U32 => 4,
            IdSize::U64 => 8,
        };

        // JVM oop size (compressed oops = 4, normal = 8) for object array sizing
        let oop_size = self.heap.oop_size as usize;
        let header_size = self.heap.header_size as i64;
        eprintln!("BFS: id_bytes={}, oop_size={}, header_size={}", id_bytes, oop_size, header_size);

        // Build class_object_id -> class_idx mapping
        let mut class_obj_to_idx: HashMap<u64, u32> = HashMap::with_capacity(self.heap.classes.len());
        // Also build class_obj_id -> class_name for static field filtering
        let mut class_obj_to_name: HashMap<u64, &str> = HashMap::with_capacity(self.heap.classes.len());
        for cls in &self.heap.classes {
            if cls.class_object_id != 0 {
                class_obj_to_idx.insert(cls.class_object_id, cls.index);
                class_obj_to_name.insert(cls.class_object_id, &cls.name);
            }
        }

        // Known (class_name, field_name) pairs whose static references should be skipped
        // because they are reference-processing chains that keep dead objects artificially alive.
        let skip_static_pairs: Vec<(&str, &str)> = vec![
            ("java/lang/ref/Finalizer", "unfinalized"),
            ("java/lang/ref/Finalizer", "queue"),
            ("java/lang/ref/FinalizerReference", "head"),
            ("java/lang/ref/FinalizerReference", "pendingNext"),
            ("sun/misc/Cleaner", "first"),
            ("jdk/internal/ref/Cleaner", "first"),
            ("jdk/internal/ref/Cleaner", "cleanerRoot"),
            ("java/lang/ref/Reference", "pending"),
            ("java/lang/ref/Reference", "discovered"),
            ("java/lang/ApplicationShutdownHooks", "hooks"),
            ("jdk/internal/misc/InnocuousThread", "INNOCUOUSTHREADGROUP"),
            ("java/util/prefs/MacOSXPreferencesFile", "timer"),
            ("java/lang/ref/ReferenceQueue", "NULL_QUEUE"),
            ("java/lang/ref/ReferenceQueue", "ENQUEUED"),
            ("java/lang/System", "lock"),
            ("java/lang/Shutdown", "hooks"),
            ("java/lang/Shutdown", "sequenceLock"),
        ];

        // Classes whose instances should be deferred during BFS.
        // Group 1: weak/finalizer/cleaner chains + ThreadGroup (skipDeferCategories in Java)
        // Group 2: collection entry classes marked isFinalizable in Java
        //   (HashMap$Node, CHM$Node, TreeMap$Entry, etc. — container descriptors mark these)
        let defer_class_names: Vec<&str> = vec![
            // skipDeferCategories
            "java/lang/ref/WeakReference",
            "java/lang/ref/SoftReference",
            "java/lang/ref/PhantomReference",
            "java/lang/ref/FinalReference",
            "java/lang/ref/Finalizer",
            "sun/misc/Cleaner",
            "jdk/internal/ref/Cleaner",
            "java/lang/ref/FinalizerReference",
            "java/util/WeakHashMap$Entry",
            "java/util/concurrent/ReferenceEntry",
            "java/lang/ThreadGroup",
            // isFinalizable: HashTableBasedMapDescriptor entry classes
            "java/util/HashMap$Node",
            "java/util/LinkedHashMap$Entry",
            "java/util/LinkedHashMap$HeadTail", // if exists
            "java/util/Hashtable$Entry",
            "java/util/concurrent/ConcurrentHashMap$Node",
            "java/util/concurrent/ConcurrentHashMap$ForwardingNode",
            "java/util/concurrent/ConcurrentHashMap$TreeBin",
            "java/util/concurrent/ConcurrentHashMap$TreeNode",
            "java/util/concurrent/ConcurrentHashMap$CounterCell",
            // isFinalizable: TreeMapDescriptor entry class
            "java/util/TreeMap$Entry",
            // isFinalizable: ConcurrentLinkedQueueDescriptor
            "java/util/concurrent/LinkedBlockingQueue$Node",
            // isFinalizable: ConcurrentHashMap.TreeBin.vals (if present)
            "java/util/concurrent/ConcurrentHashMap$ValuesView",
        ];
        let defer_class_indices: std::collections::HashSet<u32> = defer_class_names.iter()
            .filter_map(|name| self.heap.class_by_name.get(*name).copied())
            .collect();

        // For deferred (weak reference) classes, skip the "referent" field — it's a weak pointer,
        // not a strong one. Also skip "next" in finalizer/cleaner chains.
        let skip_instance_field_pairs: Vec<(&str, &str)> = vec![
            ("java/lang/ref/Reference", "referent"),
            ("java/lang/ref/Reference", "discovered"),
            ("java/lang/ref/Reference", "pendingNext"),
            ("java/lang/ref/Reference", "pending"),
            ("java/lang/ref/Finalizer", "prev"),
        ];

        // Phase 1: Build object table (obj_id -> ObjectEntry) with address_to_index
        let estimated_objects = self.heap.total_object_count as usize;
        let mut objects: Vec<ObjectEntry> = Vec::with_capacity(estimated_objects);
        let mut addr_to_idx: HashMap<u64, u32> = HashMap::with_capacity(estimated_objects);

        // Track per-class shallow sizes for instances only (kind==0)
        let mut class_inst_sizes: Vec<i64> = vec![0i64; self.heap.classes.len()];

        // Pre-compute field-based aligned body sizes (matching Java's alignedBodySize)
        // Java uses: align(headerSize + totalFieldSizes(all hierarchy), alignment)
        // rather than HPROF's instance_size which includes JVM internal padding
        let ref_sz = self.heap.oop_size as usize;
        let hdr_sz = self.heap.header_size as i64;
        let mut class_aligned_body: Vec<i64> = vec![0i64; self.heap.classes.len()];
        {
            // Build class_object_id -> index map for hierarchy walk
            let mut co_to_idx: HashMap<u64, u32> = HashMap::new();
            for cls in &self.heap.classes {
                if cls.class_object_id != 0 {
                    co_to_idx.insert(cls.class_object_id, cls.index);
                }
            }
            for cls in &self.heap.classes {
                let mut total_field_sz: i64 = 0;
                let mut current_id = cls.class_object_id;
                let mut visited = std::collections::HashSet::new();
                loop {
                    let cidx = match co_to_idx.get(&current_id) {
                        Some(&idx) => idx as usize,
                        None => break,
                    };
                    if !visited.insert(cidx) { break; }
                    if cidx >= self.heap.classes.len() { break; }
                    let c = &self.heap.classes[cidx];
                    for fd in &c.instance_fields {
                        total_field_sz += field_type_size(fd.type_code, ref_sz) as i64;
                    }
                    current_id = c.super_class_id;
                }
                let aligned = ((hdr_sz + total_field_sz + 7) & !7);
                class_aligned_body[cls.index as usize] = aligned;
            }
        }

        eprintln!("BFS Phase 1: building object table...");

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
                                    let class_idx = class_obj_to_idx
                                        .get(&inst.class_obj_id().id())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    let obj_id = inst.obj_id().id();
                                    let idx = objects.len() as u32;

                                    let header_size = self.heap.header_size as i32;
                                    let body_size = if class_idx < self.heap.classes.len() as u32 {
                                        self.heap.classes[class_idx as usize].instance_size
                                    } else {
                                        0
                                    };
                                    // Use field-based aligned size (like Java) instead of
                                    // HPROF instance_size which includes JVM internal padding
                                    let shallow = if (class_idx as usize) < class_aligned_body.len() {
                                        class_aligned_body[class_idx as usize]
                                    } else {
                                        ((header_size + body_size + 7) & !7) as i64
                                    };

                                    objects.push(ObjectEntry {
                                        obj_id,
                                        class_idx,
                                        file_offset: 0,
                                        body_size,
                                        kind: 0,
                                        array_length: 0,
                                    });

                                    addr_to_idx.insert(obj_id, idx);

                                    if (class_idx as usize) < class_inst_sizes.len() {
                                        class_inst_sizes[class_idx as usize] += shallow;
                                    }
                                }
                                Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                                    let obj_id = arr.obj_id().id();
                                    let idx = objects.len() as u32;

                                    let type_name = arr.primitive_type().java_type_name();
                                    let class_name = format!("{}[]", type_name);
                                    let class_idx = self.heap.class_by_name
                                        .get(class_name.as_str())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    let elem_size = prim_type_size(arr.primitive_type()) as i32;
                                    let length = count_prim_elements(&arr);
                                    let shallow = array_shallow_size(length, elem_size, header_size);

                                    objects.push(ObjectEntry {
                                        obj_id,
                                        class_idx,
                                        file_offset: 0,
                                        body_size: 0,
                                        kind: 1,
                                        array_length: length,
                                    });

                                    addr_to_idx.insert(obj_id, idx);
                                }
                                Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                                    let obj_id = arr.obj_id().id();
                                    let idx = objects.len() as u32;

                                    let class_idx = class_obj_to_idx
                                        .get(&arr.array_class_obj_id().id())
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    let length = arr.elements(id_size).count() as i32;
                                    let shallow = array_shallow_size(length, oop_size as i32, header_size);

                                    objects.push(ObjectEntry {
                                        obj_id,
                                        class_idx,
                                        file_offset: 0,
                                        body_size: 0,
                                        kind: 2,
                                        array_length: length,
                                    });

                                    addr_to_idx.insert(obj_id, idx);

                                }
                                // Also capture ClassDump records as objects so GC roots
                                // pointing to class objects (e.g. SystemClass roots) can resolve
                                Ok(heap_dump::SubRecord::Class(cls)) => {
                                    let class_obj_id = cls.obj_id().id();
                                    let idx = objects.len() as u32;

                                    // Find the class_idx for this class object
                                    let class_idx = class_obj_to_idx
                                        .get(&class_obj_id)
                                        .copied()
                                        .unwrap_or(u32::MAX);

                                    // java.lang.Class instance size (typically 16-24 bytes in HotSpot)
                                    let header_size = self.heap.header_size as i32;
                                    let shallow = ((header_size + 16 + 7) & !7) as i64;

                                    objects.push(ObjectEntry {
                                        obj_id: class_obj_id,
                                        class_idx,
                                        file_offset: 0,
                                        body_size: 16,
                                        kind: 3, // class object
                                        array_length: 0,
                                    });

                                    addr_to_idx.insert(class_obj_id, idx);

                                    // Don't add ClassDump sizes to per-class instance bytes —
                                    // they represent java.lang.Class instances, not regular instances.
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        eprintln!("BFS: built object table with {} objects", objects.len());

        // Phase 2: Build outgoing reference lists per object (with field info)
        eprintln!("BFS Phase 2: extracting outgoing references...");

        let mut field_pool = FieldPool::new();
        let mut outgoing_refs: Vec<Vec<OutEdge>> = vec![Vec::new(); objects.len()];
        let mut refs_found: u64 = 0;

        let class_by_address = class_obj_to_idx;

        let hprof2 = jvm_hprof::parse_hprof(data)
            .map_err(|e| anyhow::anyhow!("Failed to re-parse HPROF for refs: {:?}", e))?;

        for record in hprof2.records_iter() {
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
                                    let obj_id = inst.obj_id().id();
                                    if let Some(&obj_idx) = addr_to_idx.get(&obj_id) {
                                        let class_idx = objects[obj_idx as usize].class_idx;
                                        if class_idx < self.heap.classes.len() as u32 {
                                            let refs = extract_instance_refs(
                                                inst.fields(),
                                                &self.heap.classes[class_idx as usize],
                                                id_bytes,
                                                &addr_to_idx,
                                                &self.heap.classes,
                                                &class_by_address,
                                                &skip_instance_field_pairs,
                                                &mut field_pool,
                                            );
                                            // Filter out references targeting ClassDump objects (kind==3).
                                            // Java doesn't include ClassDump in ObjectIndex, so instance→Class
                                            // references aren't followed. Without filtering, BFS would follow
                                            // instance→ClassDump→static refs, reaching many extra objects.
                                            let refs: Vec<OutEdge> = refs.into_iter()
                                                .filter(|e| objects[e.target as usize].kind != 3)
                                                .collect();
                                            if !refs.is_empty() {
                                                refs_found += refs.len() as u64;
                                                outgoing_refs[obj_idx as usize] = refs;
                                            }
                                        }
                                    }
                                }
                                Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                                    let obj_id = arr.obj_id().id();
                                    if let Some(&obj_idx) = addr_to_idx.get(&obj_id) {
                                        let mut refs = Vec::new();
                                        for elem in arr.elements(id_size) {
                                            if let Some(Some(id)) = elem.ok() {
                                                if let Some(&child_idx) = addr_to_idx.get(&id.id()) {
                                                    // Skip ClassDump objects — Java doesn't follow
                                                    // object array elements to ClassDump targets
                                                    if objects[child_idx as usize].kind != 3 {
                                                        refs.push(OutEdge { target: child_idx, field_key: 0 });
                                                    }
                                                }
                                            }
                                        }
                                        if !refs.is_empty() {
                                            refs_found += refs.len() as u64;
                                            outgoing_refs[obj_idx as usize] = refs;
                                        }
                                    }
                                }
                                Ok(heap_dump::SubRecord::Class(cls)) => {
                                    let class_obj_id = cls.obj_id().id();
                                    let class_name = class_obj_to_name.get(&class_obj_id).copied().unwrap_or("");
                                    if let Some(&obj_idx) = addr_to_idx.get(&class_obj_id) {
                                        let mut refs = Vec::new();
                                        for sf in cls.static_fields() {
                                            if let Ok(sf) = sf {
                                                if let heap_dump::FieldValue::ObjectId(Some(id)) = sf.value() {
                                                    let addr = id.id();
                                                    if addr != 0 {
                                                        let field_name_id = sf.name_id().id();
                                                        let field_name = self.heap.string_table.get(&field_name_id)
                                                            .map(|s| s.as_str()).unwrap_or("");
                                                        let should_skip = skip_static_pairs.iter()
                                                            .any(|(cn, fn_)| *cn == class_name && *fn_ == field_name);
                                                        if !should_skip {
                                                            if let Some(&child_idx) = addr_to_idx.get(&addr) {
                                                                let fk = if field_name.is_empty() {
                                                                    1u32 // [static]
                                                                } else {
                                                                    field_pool.intern(field_name)
                                                                };
                                                                refs.push(OutEdge { target: child_idx, field_key: fk });
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        if !refs.is_empty() {
                                            refs_found += refs.len() as u64;
                                            outgoing_refs[obj_idx as usize] = refs;
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

        eprintln!("BFS: extracted {} outgoing references", refs_found);

        // Phase 3: BFS traversal from GC roots (with parent tracking for reference chains)
        eprintln!("BFS Phase 3: running BFS from {} roots...", self.heap.roots.len());

        let mut visited: Vec<bool> = vec![false; objects.len()];
        let mut deferred_queue: VecDeque<u32> = VecDeque::new();
        // Parent tracking for reference chain reconstruction
        let mut parent: Vec<u32> = vec![u32::MAX; objects.len()];
        let mut parent_edge: Vec<u32> = vec![0; objects.len()];
        // Root attribution: maps each visited object to the GC root that first reached it
        let mut root_of: Vec<u32> = vec![GC_ROOT_GARBAGE_SENTINEL; objects.len()];
        let mut root_metas: HashMap<u32, ss_stats::ref_chain::GcRootMeta> = HashMap::new();
        let mut queue: VecDeque<u32> = VecDeque::with_capacity(65536);

        // Pre-compute which classes are subclasses of java.lang.Thread
        let is_thread_subclass = {
            let mut result = vec![false; self.heap.classes.len()];
            for cls in &self.heap.classes {
                let mut current_id = cls.super_class_id;
                let mut found = false;
                let mut visited_set = std::collections::HashSet::new();
                while current_id != 0 && !found {
                    if !visited_set.insert(current_id) { break; }
                    if let Some(&super_idx) = class_by_address.get(&current_id) {
                        let super_cls = &self.heap.classes[super_idx as usize];
                        let name = super_cls.name.as_ref();
                        if name == "java/lang/Thread" || name == "java.lang.Thread" {
                            found = true;
                        }
                        current_id = super_cls.super_class_id;
                    } else {
                        break;
                    }
                }
                if found {
                    result[cls.index as usize] = true;
                }
            }
            result
        };

        // Classify root priority (matching Java's classifyRootPriority)
        // Returns: -1 = normal, 0 = drop entirely, 1 = deferred
        let classify_root_priority = |root: &HeapRootReference, obj_idx: u32| -> i32 {
            let obj = &objects[obj_idx as usize];

            if root.root_type == RootType::SystemClass {
                // Java's filterValidRootRefs drops ALL SYSTEM_CLASS roots because
                // their objectAddress is a class address (in classAddressToIndexMap).
                // Reachability is instead handled by synthetic JAVA_STATIC roots.
                return 0;
            } else {
                // Non-SystemClass roots: check the referenced object's class
                let class_idx = obj.class_idx;
                if (class_idx as usize) < self.heap.classes.len() {
                    let cls = &self.heap.classes[class_idx as usize];
                    let name = cls.name.as_ref();
                    // Priority 0: inner class, ThreadGroup, Thread subclass
                    if name.contains('$') { return 0; }
                    if name == "java/lang/ThreadGroup" || name == "java.lang.ThreadGroup" { return 0; }
                    if is_thread_subclass[class_idx as usize] { return 0; }
                    // Priority 1: isFinalizable (collection entry classes), primitive array, String, BoxedNumber
                    // Matches Java's classifyRootPriority for non-static roots
                    if obj.kind == 0 && defer_class_indices.contains(&class_idx) { return 1; }
                    if obj.kind == 1 { return 1; } // primitive array
                    if cls.category == ClassCategory::String { return 1; }
                    if cls.category == ClassCategory::BoxedNumber { return 1; }
                }
            }
            -1 // normal priority
        };

        // Seed from GC roots with priority filtering
        let mut root_count = 0u64;
        let mut unresolved_roots = 0u64;
        let mut dropped_p0 = 0u64;
        let mut deferred_p1 = 0u64;
        for root in &self.heap.roots {
            if let Some(&idx) = addr_to_idx.get(&root.object_id) {
                let priority = classify_root_priority(root, idx);
                if priority == 0 {
                    dropped_p0 += 1;
                    continue;
                }
                if !visited[idx as usize] {
                    visited[idx as usize] = true;
                    root_of[idx as usize] = idx; // Self-attribution for root objects
                    if priority == 1 {
                        deferred_queue.push_back(idx);
                        deferred_p1 += 1;
                    } else {
                        queue.push_back(idx);
                    }
                    root_count += 1;
                }
                // Accumulate root metadata (one object can have multiple root types)
                use ss_stats::ref_chain::GcRootMeta;
                root_metas.entry(idx).and_modify(|m| {
                    if !m.root_types.contains(&root.root_type) {
                        m.root_types.push(root.root_type);
                    }
                }).or_insert_with(|| GcRootMeta {
                    root_obj_idx: idx,
                    obj_id: root.object_id,
                    class_idx: objects[idx as usize].class_idx,
                    class_name: get_class_display(objects[idx as usize].class_idx, &self.heap),
                    root_types: vec![root.root_type],
                    thread_serial: root.thread_serial,
                });
            } else {
                unresolved_roots += 1;
            }
        }
        eprintln!(
            "BFS: {} roots resolved ({} dropped p0, {} deferred p1), {} unresolved out of {} total",
            root_count, dropped_p0, deferred_p1, unresolved_roots, self.heap.roots.len()
        );
        let mut visited_count: u64 = 0;
        let mut class_object_visited: u64 = 0;
        let mut deferred_count: u64 = 0;
        // Total ClassDump objects (for accurate live/garbage exclusion)
        let total_class_objects: u64 = objects.iter().filter(|o| o.kind == 3).count() as u64;

        // Java's filterValidRootRefs drops ALL JAVA_STATIC roots because their
        // objectAddress is the class's own address (in classAddressToIndexMap).
        // Java also drops ALL SYSTEM_CLASS HPROF roots (same filter).
        // So Java only uses non-SYSTEM_CLASS HPROF roots for reachability.
        // We match this by not adding synthetic JAVA_STATIC roots.
        eprintln!("BFS: no synthetic JAVA_STATIC roots (matching Java's filterValidRootRefs)");

        // Phase A: Normal BFS
        while let Some(obj_idx) = queue.pop_front() {
            visited_count += 1;

            let obj = &objects[obj_idx as usize];

            if obj.kind == 3 {
                class_object_visited += 1;
            }

            if obj.class_idx < self.heap.classes.len() as u32 {
                let header_size = self.heap.header_size as i32;
                let shallow = if obj.kind == 0 {
                    ((header_size + obj.body_size + 7) & !7) as i64
                } else if obj.kind == 1 {
                    array_shallow_size(obj.array_length, 1, header_size as i64)
                } else {
                    array_shallow_size(obj.array_length, oop_size as i32, header_size as i64)
                };

                if let Some(pc) = self.stats.get_per_class(obj.class_idx) {
                    pc.retained_size.fetch_add(shallow, Relaxed);
                }
            }

            // Enqueue children with parent tracking - defer weak/finalizer/cleaner types
            // Skip outgoing refs from ClassDump objects (kind==3) — Java doesn't traverse
            // class objects in its ObjectIndex. Static field reachability is handled by
            // the synthetic JAVA_STATIC root seeding above.
            if obj.kind != 3 {
                for edge in &outgoing_refs[obj_idx as usize] {
                    let child_idx = edge.target;
                    // Also skip edges targeting ClassDump objects — Java's filterValidRootRefs
                    // drops roots pointing at class addresses
                    if objects[child_idx as usize].kind == 3 { continue; }
                    if (child_idx as usize) < visited.len() && !visited[child_idx as usize] {
                        visited[child_idx as usize] = true;
                        parent[child_idx as usize] = obj_idx;
                        parent_edge[child_idx as usize] = edge.field_key;
                        root_of[child_idx as usize] = root_of[obj_idx as usize];
                        let child_obj = &objects[child_idx as usize];
                        if child_obj.kind == 0 && defer_class_indices.contains(&child_obj.class_idx) {
                            deferred_count += 1;
                            deferred_queue.push_back(child_idx);
                        } else {
                            queue.push_back(child_idx);
                        }
                    }
                }
            }
        }

        eprintln!(
            "BFS Phase A done: visited {}, deferred {}",
            visited_count, deferred_count
        );

        // Count visited ClassDump objects (kind==3)
        let visited_class_objs: u64 = objects.iter().enumerate()
            .filter(|(i, obj)| obj.kind == 3 && visited[*i])
            .count() as u64;
        eprintln!("BFS: {} ClassDump objects visited out of {} total class objects",
            visited_class_objs, total_class_objects);

        // Phase B: Process deferred (weak/finalizer/cleaner) objects — follow ALL outgoing refs
        // Java's second pass (isFirstPass=false) follows all children, not just deferred ones.
        // We keep processing both queues until empty (deferred can enqueue regular, and vice versa).
        while !deferred_queue.is_empty() || !queue.is_empty() {
            // Process deferred objects first (they were held back in Phase A)
            while let Some(deferred_idx) = deferred_queue.pop_front() {
                visited_count += 1;
                let obj = &objects[deferred_idx as usize];
                if obj.kind == 3 { class_object_visited += 1; }

                if obj.class_idx < self.heap.classes.len() as u32 {
                    let header_size = self.heap.header_size as i32;
                    let shallow = if obj.kind == 0 {
                        ((header_size + obj.body_size + 7) & !7) as i64
                    } else if obj.kind == 1 {
                        array_shallow_size(obj.array_length, 1, header_size as i64)
                    } else {
                        array_shallow_size(obj.array_length, oop_size as i32, header_size as i64)
                    };
                    if let Some(pc) = self.stats.get_per_class(obj.class_idx) {
                        pc.retained_size.fetch_add(shallow, Relaxed);
                    }
                }

                for edge in &outgoing_refs[deferred_idx as usize] {
                    let child_idx = edge.target;
                    if objects[child_idx as usize].kind == 3 { continue; }
                    if (child_idx as usize) < visited.len() && !visited[child_idx as usize] {
                        visited[child_idx as usize] = true;
                        parent[child_idx as usize] = deferred_idx;
                        parent_edge[child_idx as usize] = edge.field_key;
                        root_of[child_idx as usize] = root_of[deferred_idx as usize];
                        let child_obj = &objects[child_idx as usize];
                        if child_obj.kind == 0 && defer_class_indices.contains(&child_obj.class_idx) {
                            deferred_queue.push_back(child_idx);
                        } else {
                            queue.push_back(child_idx);
                        }
                    }
                }
            }

            // Drain regular queue items discovered from deferred objects
            while let Some(obj_idx) = queue.pop_front() {
                visited_count += 1;
                let obj = &objects[obj_idx as usize];
                if obj.kind == 3 { class_object_visited += 1; }

                if obj.class_idx < self.heap.classes.len() as u32 {
                    let header_size = self.heap.header_size as i32;
                    let shallow = if obj.kind == 0 {
                        ((header_size + obj.body_size + 7) & !7) as i64
                    } else if obj.kind == 1 {
                        array_shallow_size(obj.array_length, 1, header_size as i64)
                    } else {
                        array_shallow_size(obj.array_length, oop_size as i32, header_size as i64)
                    };
                    if let Some(pc) = self.stats.get_per_class(obj.class_idx) {
                        pc.retained_size.fetch_add(shallow, Relaxed);
                    }
                }

                for edge in &outgoing_refs[obj_idx as usize] {
                    let child_idx = edge.target;
                    if objects[child_idx as usize].kind == 3 { continue; }
                    if (child_idx as usize) < visited.len() && !visited[child_idx as usize] {
                        visited[child_idx as usize] = true;
                        parent[child_idx as usize] = obj_idx;
                        parent_edge[child_idx as usize] = edge.field_key;
                        root_of[child_idx as usize] = root_of[obj_idx as usize];
                        let child_obj = &objects[child_idx as usize];
                        if child_obj.kind == 0 && defer_class_indices.contains(&child_obj.class_idx) {
                            deferred_queue.push_back(child_idx);
                        } else {
                            queue.push_back(child_idx);
                        }
                    }
                }
            }
        }

        // Count excluding class objects for accurate live/garbage stats
        let total_objects = objects.len() as u64 - total_class_objects;
        let garbage_count = total_objects - (visited_count - class_object_visited);

        eprintln!(
            "BFS: visited {} / {} objects ({} garbage, {:.1}%), {} class objects excluded",
            visited_count - class_object_visited,
            total_objects,
            garbage_count,
            if total_objects > 0 { garbage_count as f64 / total_objects as f64 * 100.0 } else { 0.0 },
            total_class_objects,
        );

        // Update global stats (exclude class objects from counts)
        self.stats.total_instances.store(total_objects, Relaxed);
        self.stats.total_garbage_count.store(garbage_count, Relaxed);

        // Compute total shallow size
        let inst_total: i64 = class_inst_sizes.iter().sum();
        eprintln!("BFS: total instance shallow (field-based) = {} bytes ({} MB)", inst_total, inst_total / (1024*1024));
        let mut total_shallow: i64 = 0;
        for (class_idx, &size) in class_inst_sizes.iter().enumerate() {
            total_shallow += size;
            if let Some(pc) = self.stats.get_per_class(class_idx as u32) {
                pc.total_instance_bytes.store(size, Relaxed);
            }
        }
        // Add array bytes already accumulated by linear scan
        for pc in &self.stats.per_class {
            total_shallow += pc.array_live_bytes.load(Relaxed);
        }
        self.stats.total_shallow_size.store(total_shallow, Relaxed);

        // Compute garbage bytes with per-class breakdown (exclude class objects)
        let mut garbage_bytes: i64 = 0;
        for (i, obj) in objects.iter().enumerate() {
            if !visited[i] && obj.kind != 3 {
                let header_size = self.heap.header_size as i32;
                let shallow = if obj.kind == 0 {
                    ((header_size + obj.body_size + 7) & !7) as i64
                } else if obj.kind == 1 {
                    array_shallow_size(obj.array_length, 1, header_size as i64)
                } else {
                    array_shallow_size(obj.array_length, oop_size as i32, header_size as i64)
                };
                garbage_bytes += shallow;

                // Accumulate per-class garbage
                if let Some(pc) = self.stats.get_per_class(obj.class_idx) {
                    if obj.kind == 0 {
                        pc.instance_garbage_count.fetch_add(1, Relaxed);
                        pc.instance_garbage_bytes.fetch_add(shallow, Relaxed);
                    } else if obj.kind == 1 || obj.kind == 2 {
                        pc.array_garbage_count.fetch_add(1, Relaxed);
                        pc.array_garbage_bytes.fetch_add(shallow, Relaxed);
                    }
                }
            }
        }
        self.stats.total_garbage_bytes.store(garbage_bytes, Relaxed);

        // Per-class garbage summary (top 20)
        {
            let mut class_garb: Vec<(u64, &str)> = Vec::new();
            for pc in &self.stats.per_class {
                let ig = pc.instance_garbage_count.load(Relaxed) as u64;
                let ag = pc.array_garbage_count.load(Relaxed) as u64;
                let total_g = ig + ag;
                if total_g > 0 {
                    class_garb.push((total_g, &pc.class_name));
                }
            }
            class_garb.sort_by(|a, b| b.0.cmp(&a.0));
            eprintln!("BFS: top garbage classes:");
            for (g, name) in class_garb.iter().take(20) {
                eprintln!("  garb={}, {}", g, name);
            }
        }

        // Phase 3c: Build GC root trees for Section 4
        eprintln!("BFS Phase 3c: building GC root trees...");
        let gc_root_store = build_gc_root_trees(
            &objects,
            &root_of,
            &visited,
            &outgoing_refs,
            &root_metas,
            oop_size as i32,
            self.heap.header_size as i64,
            &self.heap,
            &field_pool,
        );
        eprintln!("BFS: built {} GC root tree entries", gc_root_store.entries.len());
        *self.stats.gc_roots.lock().unwrap() = gc_root_store;

        // Phase 4: Reference chain analysis for top classes
        eprintln!("BFS Phase 4: analyzing reference chains...");
        let ref_chain_store = analyze_ref_chains(
            data,
            &objects,
            &parent,
            &parent_edge,
            &field_pool,
            &visited,
            &addr_to_idx,
            &self.heap,
            oop_size,
            header_size,
            id_bytes,
            &outgoing_refs,
        );
        let chain_count = ref_chain_store.chains.len();

        // Phase 4b: Humongous-specific reference chain analysis for Section 19
        let mut ref_chain_store = ref_chain_store;
        let g1_region_size = self.stats.jvm_info.get().g1_region_size.max(1024 * 1024);
        let humongous_threshold = g1_region_size / 2;
        let humongous_chains = analyze_humongous_chains(
            data,
            &objects,
            &parent,
            &parent_edge,
            &field_pool,
            &visited,
            &self.heap,
            oop_size,
            header_size,
            id_bytes,
            humongous_threshold,
            &outgoing_refs,
        );
        ref_chain_store.humongous_chains = humongous_chains;

        *self.stats.ref_chains.lock().unwrap() = ref_chain_store;
        eprintln!("BFS: reference chains analyzed for {} classes", chain_count);

        Ok(TraversalResult {
            visited_count: visited_count - class_object_visited,
            total_objects,
            garbage_count,
            garbage_bytes,
            total_shallow,
        })
    }
}

/// Extract outgoing object references from an instance's field bytes.
/// Walks the full class hierarchy (this class + superclasses) since HPROF
/// stores fields in order: concrete class fields first, then superclass, etc.
fn extract_instance_refs(
    fields: &[u8],
    class_info: &ss_heap_model::class_info::JavaClassInfo,
    id_bytes: usize,
    addr_to_idx: &HashMap<u64, u32>,
    all_classes: &[ss_heap_model::class_info::JavaClassInfo],
    class_by_address: &HashMap<u64, u32>,
    skip_field_pairs: &[(&str, &str)],
    field_pool: &mut FieldPool,
) -> Vec<OutEdge> {
    let mut refs = Vec::new();
    let mut offset = 0usize;

    let mut current_class_id: Option<u64> = Some(class_info.class_object_id);

    while let Some(cid) = current_class_id {
        let cls = if cid == class_info.class_object_id {
            Some(class_info)
        } else {
            class_by_address.get(&cid).and_then(|&idx| all_classes.get(idx as usize))
        };

        if let Some(cls) = cls {
            for fd in &cls.instance_fields {
                if fd.type_code == 2 {
                    let should_skip = skip_field_pairs.iter()
                        .any(|(cn, fn_)| *cn == cls.name.as_ref() && *fn_ == fd.name.as_ref());
                    if !should_skip {
                        let addr = read_id_from(fields, offset, id_bytes);
                        if let Some(addr) = addr {
                            if addr != 0 {
                                if let Some(&child_idx) = addr_to_idx.get(&addr) {
                                    let fk = field_pool.intern(fd.name.as_ref());
                                    refs.push(OutEdge { target: child_idx, field_key: fk });
                                }
                            }
                        }
                    }
                    offset += id_bytes;
                } else {
                    offset += field_type_size(fd.type_code, id_bytes);
                }
            }
            current_class_id = if cls.super_class_id != 0 {
                Some(cls.super_class_id)
            } else {
                None
            };
        } else {
            break;
        }
    }

    refs
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

fn field_type_size(type_code: u8, id_bytes: usize) -> usize {
    match type_code {
        2 => id_bytes, // object
        4 => 1,        // boolean
        5 => 2,        // char
        6 => 4,        // float
        7 => 8,        // double
        8 => 1,        // byte
        9 => 2,        // short
        10 => 4,       // int
        11 => 8,       // long
        _ => 0,
    }
}

fn prim_type_size(pt: heap_dump::PrimitiveArrayType) -> usize {
    match pt {
        heap_dump::PrimitiveArrayType::Boolean => 1,
        heap_dump::PrimitiveArrayType::Byte => 1,
        heap_dump::PrimitiveArrayType::Char => 2,
        heap_dump::PrimitiveArrayType::Short => 2,
        heap_dump::PrimitiveArrayType::Float => 4,
        heap_dump::PrimitiveArrayType::Int => 4,
        heap_dump::PrimitiveArrayType::Double => 8,
        heap_dump::PrimitiveArrayType::Long => 8,
    }
}

fn count_prim_elements(arr: &heap_dump::PrimitiveArray) -> i32 {
    if let Some(it) = arr.booleans() { it.count() as i32 }
    else if let Some(it) = arr.bytes() { it.count() as i32 }
    else if let Some(it) = arr.chars() { it.count() as i32 }
    else if let Some(it) = arr.shorts() { it.count() as i32 }
    else if let Some(it) = arr.ints() { it.count() as i32 }
    else if let Some(it) = arr.longs() { it.count() as i32 }
    else if let Some(it) = arr.floats() { it.count() as i32 }
    else if let Some(it) = arr.doubles() { it.count() as i32 }
    else { 0 }
}

fn array_shallow_size(length: i32, element_size: i32, header_size: i64) -> i64 {
    let data_size = length as i64 * element_size as i64;
    let raw = header_size + data_size;
    (raw + 7) & !7
}

pub struct TraversalResult {
    pub visited_count: u64,
    pub total_objects: u64,
    pub garbage_count: u64,
    pub garbage_bytes: i64,
    pub total_shallow: i64,
}

/// Analyze reference chains for the top classes by shallow size.
fn analyze_ref_chains(
    data: &[u8],
    objects: &[ObjectEntry],
    parent: &[u32],
    parent_edge: &[u32],
    field_pool: &FieldPool,
    visited: &[bool],
    addr_to_idx: &HashMap<u64, u32>,
    heap: &HeapDump,
    oop_size: usize,
    header_size: i64,
    id_bytes: usize,
    outgoing_refs: &[Vec<OutEdge>],
) -> RefChainStore {
    // Find top 50 classes by total_instance_bytes
    let mut class_sizes: Vec<(u32, i64)> = Vec::new();
    for cls in &heap.classes {
        let idx = cls.index;
        let mut total: i64 = 0;
        for (i, obj) in objects.iter().enumerate() {
            if obj.class_idx == idx && visited[i] && obj.kind != 3 {
                total += compute_obj_shallow(obj, oop_size as i32, header_size);
            }
        }
        if total > 0 {
            class_sizes.push((idx, total));
        }
    }
    class_sizes.sort_by(|a, b| b.1.cmp(&a.1));
    let top_classes: Vec<u32> = class_sizes.iter().take(50).map(|(idx, _)| *idx).collect();

    // Collect all sampled object indices across all top classes
    // Use "effective size" for sorting: for instances (e.g. String), include the size
    // of referenced objects via outgoing_refs. This ensures large-content Strings
    // (e.g. SQL statements backed by big byte[]) rank higher than tiny class-name Strings.
    let string_class_idx = heap.class_by_name.get("java/lang/String").copied();
    let mut all_sampled_indices: Vec<u32> = Vec::new();
    let mut per_class_samples: Vec<(u32, Vec<u32>, Vec<u32>)> = Vec::new(); // (class_idx, live_objs, samples)

    for &class_idx in &top_classes {
        let mut live_objs: Vec<(i64, u32)> = Vec::new(); // (effective_size, obj_index)
        for (i, obj) in objects.iter().enumerate() {
            if obj.class_idx == class_idx && visited[i] && obj.kind != 3 {
                let mut sz = compute_obj_shallow(obj, oop_size as i32, header_size);
                // For instance objects (especially String), add referenced children's shallow sizes
                // to get a more meaningful ranking. This is similar to impl-inclusive size.
                if obj.kind == 0 {
                    for edge in &outgoing_refs[i] {
                        let child = &objects[edge.target as usize];
                        if child.kind == 1 || child.kind == 2 {
                            // Only add array children (the main data holders)
                            sz += compute_obj_shallow(child, oop_size as i32, header_size);
                        }
                    }
                }
                live_objs.push((sz, i as u32));
            }
        }
        if live_objs.is_empty() {
            continue;
        }

        // Sort by effective size descending
        live_objs.sort_by(|a, b| b.0.cmp(&a.0));
        let top_by_size: Vec<u32> = live_objs.iter().take(100).map(|(_, idx)| *idx).collect();

        // Random portion: use deterministic step sampling from the full set
        let step = (live_objs.len() / 100).max(1);
        let random_samples: Vec<u32> = live_objs.iter().step_by(step).take(100).map(|(_, idx)| *idx).collect();

        // Merge and deduplicate
        let mut seen: HashMap<u32, ()> = HashMap::new();
        let mut samples: Vec<u32> = Vec::new();
        for &idx in top_by_size.iter().chain(random_samples.iter()) {
            if !seen.contains_key(&idx) {
                seen.insert(idx, ());
                samples.push(idx);
            }
        }
        samples.truncate(200);

        let live_obj_indices: Vec<u32> = live_objs.iter().map(|(_, idx)| *idx).collect();
        all_sampled_indices.extend_from_slice(&samples);
        per_class_samples.push((class_idx, live_obj_indices, samples));
    }

    // Build rich descriptions for all sampled objects via targeted HPROF re-parse
    let descriptions = build_sample_descriptions(data, &all_sampled_indices, objects, id_bytes, heap);

    let mut store = RefChainStore::default();

    for (class_idx, live_objs, samples) in &per_class_samples {
        let mut groups: HashMap<Vec<(u32, u32)>, Vec<u32>> = HashMap::new();
        for &obj_idx in samples {
            let chain = trace_chain(obj_idx, parent, parent_edge, objects);
            groups.entry(chain).or_default().push(obj_idx);
        }

        let mut class_chains = ClassRefChains::default();

        for (path_keys, obj_indices) in &groups {
            let total_bytes: i64 = obj_indices.iter()
                .map(|&idx| compute_obj_shallow(&objects[idx as usize], oop_size as i32, header_size))
                .sum();
            let sample_count_in_group = obj_indices.len() as u64;

            let scale = if sample_count_in_group > 0 {
                live_objs.len() as f64 / samples.len() as f64
            } else {
                1.0
            };
            let extrapolated_bytes = (total_bytes as f64 * scale) as i64;
            let extrapolated_count = (sample_count_in_group as f64 * scale) as u64;

            let path: Vec<ChainStep> = path_keys.iter().rev().map(|&(cls_idx, fld_key)| {
                let class_name = if (cls_idx as usize) < heap.classes.len() {
                    heap.classes[cls_idx as usize].name.replace('/', ".")
                } else {
                    "unknown".to_string()
                };
                let field_name = field_pool.get(fld_key).to_string();
                if field_name == "[elements]" {
                    ChainStep::ArrayElement { class_name }
                } else if field_name == "[static]" {
                    ChainStep::Root { description: format!("Java Static [{}]", class_name) }
                } else {
                    ChainStep::Field { class_name, field_name }
                }
            }).collect();

            // Build sample descriptions using rich content from HPROF re-parse
            let sample_strs: Vec<String> = obj_indices.iter().take(5)
                .map(|&idx| {
                    descriptions.get(&idx).cloned()
                        .unwrap_or_else(|| describe_object_simple(idx, objects, heap))
                })
                .collect();

            let group = RefChainGroup {
                size_bytes: extrapolated_bytes,
                count: extrapolated_count,
                samples: sample_strs,
                path,
            };

            if path_keys.len() <= 2 {
                class_chains.expensive_fields.push(group);
            } else {
                class_chains.full_chains.push(group);
            }
        }

        class_chains.expensive_fields.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        class_chains.full_chains.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        class_chains.expensive_fields.truncate(10);
        class_chains.full_chains.truncate(10);

        store.chains.insert(*class_idx, class_chains);
    }

    store
}

/// Analyze reference chains specifically for humongous objects (>= half G1 region size).
/// Returns ClassRefChains with expensive_fields and full_chains for Section 19.
fn analyze_humongous_chains(
    data: &[u8],
    objects: &[ObjectEntry],
    parent: &[u32],
    parent_edge: &[u32],
    field_pool: &FieldPool,
    visited: &[bool],
    heap: &HeapDump,
    oop_size: usize,
    header_size: i64,
    id_bytes: usize,
    humongous_threshold: i64,
    outgoing_refs: &[Vec<OutEdge>],
) -> ClassRefChains {
    // Collect all humongous objects (both live and garbage)
    let mut humongous_objs: Vec<u32> = Vec::new();
    for (i, obj) in objects.iter().enumerate() {
        if obj.kind == 3 { continue; }
        let size = compute_obj_shallow(obj, oop_size as i32, header_size);
        if size >= humongous_threshold {
            humongous_objs.push(i as u32);
        }
    }

    if humongous_objs.is_empty() {
        return ClassRefChains::default();
    }

    // Build rich descriptions for humongous objects
    let descriptions = build_sample_descriptions(data, &humongous_objs, objects, id_bytes, heap);

    // Build full reverse ref map for ALL objects so chain tracing can walk through
    // non-humongous intermediaries (e.g. String.value → Searcher → byte[])
    let mut reverse_refs: HashMap<u32, Vec<(u32, u32)>> = HashMap::new(); // target -> [(source, field_key)]
    for (src_idx, refs) in outgoing_refs.iter().enumerate() {
        for edge in refs {
            reverse_refs.entry(edge.target).or_default().push((src_idx as u32, edge.field_key));
        }
    }

    // Trace chains using BFS parent for visited objects, reverse ref for garbage
    let mut groups: HashMap<Vec<(u32, u32)>, Vec<u32>> = HashMap::new();
    for &obj_idx in &humongous_objs {
        let chain = trace_humongous_chain(obj_idx, parent, parent_edge, objects, &reverse_refs);
        groups.entry(chain).or_default().push(obj_idx);
    }

    let mut class_chains = ClassRefChains::default();

    for (path_keys, obj_indices) in &groups {
        let total_bytes: i64 = obj_indices.iter()
            .map(|&idx| compute_obj_shallow(&objects[idx as usize], oop_size as i32, header_size))
            .sum();
        let count = obj_indices.len() as u64;

        // Path goes from closest-to-object (first step) toward root (last step),
        // matching Java's rendering order: no reversal needed.
        let path: Vec<ChainStep> = path_keys.iter().map(|&(cls_idx, fld_key)| {
            // Special "Unreachable" marker
            if cls_idx == u32::MAX {
                return ChainStep::Root { description: "Unreachable".to_string() };
            }
            let class_name = if (cls_idx as usize) < heap.classes.len() {
                heap.classes[cls_idx as usize].name.replace('/', ".")
            } else {
                "unknown".to_string()
            };
            let field_name = field_pool.get(fld_key).to_string();
            if field_name == "[elements]" {
                ChainStep::ArrayElement { class_name }
            } else if field_name == "[static]" {
                ChainStep::Root { description: format!("Java Static [{}]", class_name) }
            } else {
                ChainStep::Field { class_name, field_name }
            }
        }).collect();

        let sample_strs: Vec<String> = obj_indices.iter().take(5)
            .map(|&idx| {
                descriptions.get(&idx).cloned()
                    .unwrap_or_else(|| describe_object_simple(idx, objects, heap))
            })
            .collect();

        let group = RefChainGroup {
            size_bytes: total_bytes,
            count,
            samples: sample_strs,
            path: path.clone(),
        };

        // Java puts the same group in BOTH panels:
        // - Expensive data fields: shows only the first path step
        // - Full reference chains: shows all steps
        class_chains.expensive_fields.push(group.clone());
        class_chains.full_chains.push(group);
    }

    class_chains.expensive_fields.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    class_chains.full_chains.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

    class_chains
}

/// Re-parse HPROF to build rich sample descriptions for sampled objects.
fn build_sample_descriptions(
    data: &[u8],
    sampled_indices: &[u32],
    objects: &[ObjectEntry],
    id_bytes: usize,
    heap: &HeapDump,
) -> HashMap<u32, String> {
    let sampled_ids: HashMap<u64, u32> = sampled_indices.iter()
        .map(|&idx| (objects[idx as usize].obj_id, idx))
        .collect();

    let string_class_idx = heap.class_by_name.get("java/lang/String").copied();

    // Find String.value field offset by looking at instance_fields
    let string_value_offset: Option<(usize, usize)> = string_class_idx.and_then(|cls_idx| {
        let cls = &heap.classes[cls_idx as usize];
        let mut offset = 0usize;
        for fd in &cls.instance_fields {
            if fd.name.as_ref() == "value" && fd.type_code == 2 {
                return Some((offset, id_bytes));
            }
            offset += field_type_size(fd.type_code, id_bytes);
        }
        None
    });

    // --- Pass 1: collect String value array IDs ---
    let mut string_value_targets: HashMap<u64, u32> = HashMap::new(); // array_obj_id -> string_idx
    {
        let hprof = match jvm_hprof::parse_hprof(data) {
            Ok(h) => h,
            Err(_) => return HashMap::new(),
        };
        for record in hprof.records_iter() {
            let record = match record {
                Ok(r) => r,
                Err(_) => continue,
            };
            if matches!(record.tag(), RecordTag::HeapDump | RecordTag::HeapDumpSegment) {
                if let Some(Ok(seg)) = record.as_heap_dump_segment() {
                    for sr in seg.sub_records() {
                        if let Ok(heap_dump::SubRecord::Instance(inst)) = sr {
                            let obj_id = inst.obj_id().id();
                            if let Some(&idx) = sampled_ids.get(&obj_id) {
                                let obj = &objects[idx as usize];
                                if string_class_idx == Some(obj.class_idx) {
                                    if let Some((offset, field_size)) = string_value_offset {
                                        if let Some(arr_id) = read_id_from(inst.fields(), offset, field_size) {
                                            string_value_targets.insert(arr_id, idx);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Pass 2: collect descriptions and bytearray content ---
    let hprof = match jvm_hprof::parse_hprof(data) {
        Ok(h) => h,
        Err(_) => return HashMap::new(),
    };
    let id_size = hprof.header().id_size();
    let mut results: HashMap<u32, String> = HashMap::new();
    let mut bytearray_content: HashMap<u64, Vec<u8>> = HashMap::new();
    let mut string_values: Vec<(u32, u64)> = Vec::new();

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
                            Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                                let obj_id = arr.obj_id().id();
                                let is_sampled = sampled_ids.contains_key(&obj_id);
                                let is_string_value = string_value_targets.contains_key(&obj_id);

                                if is_sampled || is_string_value {
                                    let arr_len = count_prim_elements(&arr);
                                    // Cache byte content for String resolution (limit size)
                                    if arr_len <= 65536 {
                                        let bytes = collect_prim_bytes(&arr);
                                        bytearray_content.insert(obj_id, bytes);
                                    }

                                    if let Some(&idx) = sampled_ids.get(&obj_id) {
                                        let obj = &objects[idx as usize];
                                        let base_type = arr.primitive_type().java_type_name();
                                        let desc = format_prim_array_sample(&arr, base_type, obj.array_length);
                                        results.insert(idx, desc);
                                    }
                                }
                            }
                            Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                                let obj_id = arr.obj_id().id();
                                if let Some(&idx) = sampled_ids.get(&obj_id) {
                                    let obj = &objects[idx as usize];
                                    let class_name = get_class_display(obj.class_idx, heap);
                                    let mut elem_descs: Vec<String> = Vec::new();
                                    for elem in arr.elements(id_size).take(5) {
                                        if let Some(id) = elem.ok().flatten() {
                                            let addr = id.id();
                                            if addr != 0 {
                                                if let Some(&elem_class_idx) = heap.object_to_class_idx.get(&addr) {
                                                    let elem_name = get_class_display(elem_class_idx, heap);
                                                    if elem_name == "java.lang.String" {
                                                        elem_descs.push(format!("<t-c>String</t-c>@{:x}", addr));
                                                    } else {
                                                        elem_descs.push(format!("<t-c>{}</t-c>", elem_name));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if elem_descs.is_empty() {
                                        results.insert(idx, format!("<t-c>{}</t-c>[{}]", class_name, format_number(obj.array_length as i64)));
                                    } else {
                                        let trailing = if obj.array_length > 5 { format!(", ... ({} more)", obj.array_length - 5) } else { String::new() };
                                        results.insert(idx, format!("<t-c>{}</t-c>[{}]{{<t-s>{}{}</t-s>}}",
                                            class_name, format_number(obj.array_length as i64), elem_descs.join(", "), trailing));
                                    }
                                }
                            }
                            Ok(heap_dump::SubRecord::Instance(inst)) => {
                                let obj_id = inst.obj_id().id();
                                if let Some(&idx) = sampled_ids.get(&obj_id) {
                                    let obj = &objects[idx as usize];
                                    if string_class_idx == Some(obj.class_idx) {
                                        if let Some((offset, field_size)) = string_value_offset {
                                            if let Some(arr_id) = read_id_from(inst.fields(), offset, field_size) {
                                                string_values.push((idx, arr_id));
                                            }
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

    // Resolve String content using collected byte[] data
    for (string_idx, value_array_id) in string_values {
        if let Some(bytes) = bytearray_content.get(&value_array_id) {
            let text = String::from_utf8_lossy(bytes);
            let truncated: String = text.chars().take(200).collect();
            if !truncated.is_empty() {
                results.insert(string_idx, format!("<t-c>String</t-c> \"{}\"", escape_html(&truncated)));
            }
        }
    }

    // Fill in defaults for any sampled objects not yet described
    for &idx in sampled_indices {
        if !results.contains_key(&idx) {
            results.insert(idx, describe_object_simple(idx, objects, heap));
        }
    }

    results
}

/// Format a primitive array sample: `<t-c>byte</t-c>[16,384]{<t-s>1, 1, 0, 'S', ...</t-s> (13,047 trailing 0s)}`
fn format_prim_array_sample(arr: &heap_dump::PrimitiveArray, base_type: &str, length: i32) -> String {
    let max_show = 16;
    let mut elements: Vec<String> = Vec::new();
    let mut trailing_zeros: i32 = 0;
    let mut all_values: Vec<i64> = Vec::new();

    // Collect all values (up to a reasonable limit for trailing zero count)
    let count_limit = length.min(65536) as usize;

    match arr.primitive_type() {
        heap_dump::PrimitiveArrayType::Boolean => {
            if let Some(it) = arr.booleans() {
                for v in it.take(count_limit).flatten() { all_values.push(if v { 1 } else { 0 }); }
            }
        }
        heap_dump::PrimitiveArrayType::Byte => {
            if let Some(it) = arr.bytes() {
                for v in it.take(count_limit).flatten() { all_values.push(v as i64); }
            }
        }
        heap_dump::PrimitiveArrayType::Char => {
            if let Some(it) = arr.chars() {
                for v in it.take(count_limit).flatten() { all_values.push(v as i64); }
            }
        }
        heap_dump::PrimitiveArrayType::Short => {
            if let Some(it) = arr.shorts() {
                for v in it.take(count_limit).flatten() { all_values.push(v as i64); }
            }
        }
        heap_dump::PrimitiveArrayType::Float => {
            if let Some(it) = arr.floats() {
                for v in it.take(count_limit).flatten() {
                    let bits = v.to_bits() as i64;
                    all_values.push(bits);
                }
            }
        }
        heap_dump::PrimitiveArrayType::Int => {
            if let Some(it) = arr.ints() {
                for v in it.take(count_limit).flatten() { all_values.push(v as i64); }
            }
        }
        heap_dump::PrimitiveArrayType::Double => {
            if let Some(it) = arr.doubles() {
                for v in it.take(count_limit).flatten() {
                    let bits = v.to_bits() as i64;
                    all_values.push(bits);
                }
            }
        }
        heap_dump::PrimitiveArrayType::Long => {
            if let Some(it) = arr.longs() {
                for v in it.take(count_limit).flatten() { all_values.push(v); }
            }
        }
    }

    // Count trailing zeros
    for &v in all_values.iter().rev() {
        if v == 0 {
            trailing_zeros += 1;
        } else {
            break;
        }
    }

    // Format first max_show elements
    let is_byte = matches!(arr.primitive_type(), heap_dump::PrimitiveArrayType::Byte);
    let is_char = matches!(arr.primitive_type(), heap_dump::PrimitiveArrayType::Char);

    for v in all_values.iter().take(max_show) {
        if is_byte {
            let b = *v as u8;
            if b >= 32 && b <= 126 {
                elements.push(format!("'{}'", b as char));
            } else {
                elements.push(v.to_string());
            }
        } else if is_char {
            let c = *v as u16;
            if c >= 32 && c < 0xD800 {
                elements.push(format!("'{}'", char::from_u32(c as u32).unwrap_or('?')));
            } else {
                elements.push(v.to_string());
            }
        } else {
            elements.push(v.to_string());
        }
    }

    let length_str = format_number(length as i64);
    let elem_str = elements.join(", ");
    let has_more = all_values.len() > max_show || length as usize > count_limit;

    let trailing_str = if trailing_zeros > 0 {
        format!(" ({},{} trailing 0s)", format_number(trailing_zeros as i64), trailing_zeros)
    } else if has_more {
        ", ...".to_string()
    } else {
        String::new()
    };

    // Compute non-zero content length for "more" indicator
    let non_zero_count = all_values.len() - trailing_zeros as usize;
    let more_before_trailing = if non_zero_count > max_show {
        format!(", ... ({} elements omitted)", non_zero_count - max_show)
    } else {
        String::new()
    };

    if trailing_zeros > 0 {
        // Show elements, then indicate trailing zeros
        let shown_nonzero = elements.len().min(non_zero_count as usize);
        let shown_elems: String = elements[..shown_nonzero].join(", ");
        let tz_str = if length > count_limit as i32 {
            // Estimate real trailing zeros from sample
            format!("{},{}", format_number(trailing_zeros as i64), trailing_zeros)
        } else {
            format_number(trailing_zeros as i64)
        };
        let trailing_suffix = format!("... ({} trailing 0s)", tz_str);
        format!("<t-c>{}</t-c>[{}]{{<t-s>{}, {}</t-s>}}", base_type, length_str, shown_elems, trailing_suffix)
    } else {
        let suffix = if has_more { ", ..." } else { "" };
        format!("<t-c>{}</t-c>[{}]{{<t-s>{}{}</t-s>}}", base_type, length_str, elem_str, suffix)
    }
}

/// Collect raw bytes from a primitive array (for byte[] content resolution).
fn collect_prim_bytes(arr: &heap_dump::PrimitiveArray) -> Vec<u8> {
    if let Some(it) = arr.bytes() {
        it.flatten().map(|b| b as u8).collect()
    } else if let Some(it) = arr.booleans() {
        it.flatten().map(|b| if b { 1u8 } else { 0u8 }).collect()
    } else if let Some(it) = arr.chars() {
        let chars: Vec<u16> = it.flatten().collect();
        chars.iter().flat_map(|c| c.to_be_bytes()).collect()
    } else if let Some(it) = arr.shorts() {
        let shorts: Vec<i16> = it.flatten().collect();
        shorts.iter().flat_map(|s| s.to_be_bytes()).collect()
    } else if let Some(it) = arr.ints() {
        let ints: Vec<i32> = it.take(4096).flatten().collect();
        ints.iter().flat_map(|i| i.to_be_bytes()).collect()
    } else {
        Vec::new()
    }
}

/// Get class display name (dotted).
fn get_class_display(class_idx: u32, heap: &HeapDump) -> String {
    if (class_idx as usize) < heap.classes.len() {
        heap.classes[class_idx as usize].name.replace('/', ".")
    } else {
        "unknown".to_string()
    }
}

/// Format number with comma separators.
fn format_number(n: i64) -> String {
    let s = n.abs().to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    let mut chars: Vec<char> = s.chars().collect();
    chars.reverse();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    let formatted: String = result.chars().rev().collect();
    if n < 0 { format!("-{}", formatted) } else { formatted }
}

/// Escape HTML special characters.
fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Trace from an object up to root via parent[], returning (parent_class_idx, field_key) pairs.
fn trace_chain(start: u32, parent: &[u32], parent_edge: &[u32], objects: &[ObjectEntry]) -> Vec<(u32, u32)> {
    let mut chain = Vec::new();
    let mut current = start;
    for _ in 0..60 {
        let p = parent[current as usize];
        if p == u32::MAX {
            break;
        }
        let parent_class = objects[p as usize].class_idx;
        let edge_key = parent_edge[current as usize];
        chain.push((parent_class, edge_key));
        current = p;
    }
    chain
}

/// Trace chain for humongous objects, handling both visited (BFS parent) and
/// garbage (reverse ref lookup) objects. Returns (parent_class_idx, field_key) pairs
/// from the humongous object upward. If the chain reaches a garbage object without
/// further parents, appends a "virtual root" marker.
fn trace_humongous_chain(
    start: u32,
    parent: &[u32],
    parent_edge: &[u32],
    objects: &[ObjectEntry],
    reverse_refs: &HashMap<u32, Vec<(u32, u32)>>,
) -> Vec<(u32, u32)> {
    let mut chain = Vec::new();
    let mut current = start;

    for _ in 0..60 {
        let p = parent[current as usize];
        if p != u32::MAX {
            // Has BFS parent — normal case
            let parent_class = objects[p as usize].class_idx;
            let edge_key = parent_edge[current as usize];
            chain.push((parent_class, edge_key));
            current = p;
        } else {
            // No BFS parent — try reverse ref lookup (garbage object)
            if let Some(referrers) = reverse_refs.get(&current) {
                if let Some(&(src_idx, field_key)) = referrers.first() {
                    let src_class = objects[src_idx as usize].class_idx;
                    chain.push((src_class, field_key));
                    current = src_idx;
                    continue;
                }
            }
            // No parent and no reverse ref — truly unreachable
            // Use a special marker for "Unreachable" (class_idx = u32::MAX)
            chain.push((u32::MAX, u32::MAX));
            break;
        }
    }
    chain
}

/// Compute shallow size of an object entry.
fn compute_obj_shallow(obj: &ObjectEntry, oop_size: i32, header_size: i64) -> i64 {
    match obj.kind {
        0 => (header_size + obj.body_size as i64 + 7) & !7,
        1 => array_shallow_size(obj.array_length, 1, header_size),
        2 => array_shallow_size(obj.array_length, oop_size, header_size),
        _ => 0,
    }
}

/// Simple object description fallback (no content).
fn describe_object_simple(obj_idx: u32, objects: &[ObjectEntry], heap: &HeapDump) -> String {
    let obj = &objects[obj_idx as usize];
    let class_name = get_class_display(obj.class_idx, heap);
    match obj.kind {
        1 | 2 => format!("<t-c>{}</t-c>[{}]", class_name, format_number(obj.array_length as i64)),
        _ => format!("<t-c>{}</t-c>", class_name),
    }
}

/// Check if a class name matches Java's skipClassFieldPairs list.
/// These classes have static fields that hold references to reference-processing
/// infrastructure (Finalizer, Cleaner, etc.) and should be deferred in BFS.
fn matches_skip_class_pair(name: &str) -> bool {
    matches!(name,
        "java/lang/ref/Finalizer"
        | "java.lang.ref.Finalizer"
        | "java/lang/ref/FinalizerReference"
        | "java.lang.ref.FinalizerReference"
        | "sun/misc/Cleaner"
        | "sun.misc.Cleaner"
        | "jdk/internal/ref/Cleaner"
        | "jdk.internal.ref.Cleaner"
        | "java/lang/ref/Reference"
        | "java.lang.ref.Reference"
        | "java/lang/ApplicationShutdownHooks"
        | "java.lang.ApplicationShutdownHooks"
        | "jdk/internal/misc/InnocuousThread"
        | "jdk.internal.misc.InnocuousThread"
        | "java/util/prefs/MacOSXPreferencesFile"
        | "java.util.prefs.MacOSXPreferencesFile"
    )
}

// ============================================================
// GC Root tree building for Section 4
// ============================================================

/// Sentinel value in root_of[] marking unvisited/garbage objects.
const GC_ROOT_GARBAGE_SENTINEL: u32 = u32::MAX - 1;

const GC_ROOT_MAX_DEPTH: usize = 8;
const GC_ROOT_MAX_CHILDREN: usize = 10;
const GC_ROOT_TOP_N: usize = 20;
const GC_ROOT_SAMPLE_N: usize = 40;

fn build_gc_root_trees(
    objects: &[ObjectEntry],
    root_of: &[u32],
    visited: &[bool],
    outgoing_refs: &[Vec<OutEdge>],
    root_metas: &HashMap<u32, ss_stats::ref_chain::GcRootMeta>,
    oop_size: i32,
    header_size: i64,
    heap: &HeapDump,
    field_pool: &FieldPool,
) -> ss_stats::ref_chain::GcRootStore {
    // Step 1: Aggregate per-root retained sizes
    let mut root_retained: HashMap<u32, (i64, u64)> = HashMap::new();
    let mut garbage_bytes: i64 = 0;
    let mut garbage_count: u64 = 0;

    for (idx, obj) in objects.iter().enumerate() {
        if obj.kind == 3 { continue; }
        let r = root_of[idx];
        if r == GC_ROOT_GARBAGE_SENTINEL {
            if !visited[idx] {
                garbage_bytes += compute_obj_shallow(obj, oop_size, header_size);
                garbage_count += 1;
            }
            continue;
        }
        let shallow = compute_obj_shallow(obj, oop_size, header_size);
        let entry = root_retained.entry(r).or_insert((0, 0));
        entry.0 += shallow;
        entry.1 += 1;
    }

    // Step 2: Sort roots by retained size descending
    let mut sorted_roots: Vec<(u32, i64, u64)> = root_retained.into_iter()
        .map(|(k, (b, c))| (k, b, c))
        .collect();
    sorted_roots.sort_by(|a, b| b.1.cmp(&a.1));

    // Step 3: Build trees for top N roots
    let top_n = GC_ROOT_TOP_N.min(sorted_roots.len());
    let mut entries: Vec<ss_stats::ref_chain::GcRootEntry> = Vec::with_capacity(top_n);

    for &(root_idx, retained_bytes, retained_count) in &sorted_roots[..top_n] {
        let meta = root_metas.get(&root_idx).cloned().unwrap_or_else(|| {
            ss_stats::ref_chain::GcRootMeta {
                root_obj_idx: root_idx,
                obj_id: objects[root_idx as usize].obj_id,
                class_idx: objects[root_idx as usize].class_idx,
                class_name: get_class_display(objects[root_idx as usize].class_idx, heap),
                root_types: vec![ss_heap_model::root::RootType::Unknown],
                thread_serial: None,
            }
        });

        let self_bytes = compute_obj_shallow(&objects[root_idx as usize], oop_size, header_size);

        let tree = build_single_root_tree(
            root_idx,
            root_idx,
            objects,
            root_of,
            outgoing_refs,
            oop_size,
            header_size,
            heap,
            field_pool,
            0,
            retained_bytes,
        );

        entries.push(ss_stats::ref_chain::GcRootEntry {
            meta,
            retained_bytes,
            retained_count,
            self_bytes,
            tree,
        });
    }

    // Step 4: Compute remaining roots summary
    let remaining = if sorted_roots.len() > top_n { &sorted_roots[top_n..] } else { &[] };
    let remaining_count: u64 = remaining.iter().map(|(_, _, c)| *c).sum();
    let remaining_bytes: i64 = remaining.iter().map(|(_, b, _)| *b).sum();

    // Random sample of remaining roots
    let sample_n = GC_ROOT_SAMPLE_N.min(remaining.len());
    let step = if remaining.len() > sample_n { remaining.len() / sample_n } else { 1 };
    let remaining_samples: Vec<ss_stats::ref_chain::GcRootMeta> = remaining.iter().step_by(step)
        .take(GC_ROOT_SAMPLE_N)
        .filter_map(|&(root_idx, _, _)| root_metas.get(&root_idx).cloned())
        .collect();

    // Step 5: Build garbage info
    let mut garbage_class_map: HashMap<u32, (u64, i64)> = HashMap::new();
    for (idx, obj) in objects.iter().enumerate() {
        if obj.kind == 3 { continue; }
        if !visited[idx] && root_of[idx] == GC_ROOT_GARBAGE_SENTINEL {
            let shallow = compute_obj_shallow(obj, oop_size, header_size);
            let entry = garbage_class_map.entry(obj.class_idx).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += shallow;
        }
    }
    let mut garbage_top: Vec<(String, u64, i64)> = garbage_class_map.into_iter()
        .map(|(cls_idx, (cnt, bytes))| (get_class_display(cls_idx, heap), cnt, bytes))
        .collect();
    garbage_top.sort_by(|a, b| b.2.cmp(&a.2));

    ss_stats::ref_chain::GcRootStore {
        entries,
        remaining_count,
        remaining_bytes,
        remaining_samples,
        garbage: ss_stats::ref_chain::GarbageRootInfo {
            count: garbage_count,
            bytes: garbage_bytes,
            top_classes: garbage_top.into_iter().take(20).collect(),
        },
    }
}

fn build_single_root_tree(
    current_idx: u32,
    root_idx: u32,
    objects: &[ObjectEntry],
    root_of: &[u32],
    outgoing_refs: &[Vec<OutEdge>],
    oop_size: i32,
    header_size: i64,
    heap: &HeapDump,
    field_pool: &FieldPool,
    depth: usize,
    root_retained_bytes: i64,
) -> ss_stats::ref_chain::GcRootTreeNode {
    let obj = &objects[current_idx as usize];
    let self_bytes = compute_obj_shallow(obj, oop_size, header_size);

    let mut node = ss_stats::ref_chain::GcRootTreeNode {
        field_name: None,
        class_name: get_class_display(obj.class_idx, heap),
        self_count: 1,
        self_bytes,
        retained_bytes: self_bytes,
        retained_count: 1,
        children: Vec::new(),
        collapsed_count: 0,
        collapsed_bytes: 0,
    };

    if depth >= GC_ROOT_MAX_DEPTH {
        return node;
    }

    // Group outgoing refs by (field_key, target_class_idx)
    let mut child_groups: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for edge in &outgoing_refs[current_idx as usize] {
        let child_idx = edge.target;
        if (child_idx as usize) >= objects.len() { continue; }
        if objects[child_idx as usize].kind == 3 { continue; }
        if root_of[child_idx as usize] != root_idx { continue; }
        let child_class = objects[child_idx as usize].class_idx;
        child_groups.entry((edge.field_key, child_class)).or_default().push(child_idx);
    }

    // Sort groups by total shallow size descending
    let mut groups: Vec<((u32, u32), Vec<u32>)> = child_groups.into_iter().collect();
    groups.sort_by(|a, b| {
        let sa: i64 = a.1.iter().map(|&idx| compute_obj_shallow(&objects[idx as usize], oop_size, header_size)).sum();
        let sb: i64 = b.1.iter().map(|&idx| compute_obj_shallow(&objects[idx as usize], oop_size, header_size)).sum();
        sb.cmp(&sa)
    });

    // Prune threshold: don't prune at shallow depths (direct field refs are always small),
    // then skip groups smaller than 0.01% of root retained (min 256 bytes) at deeper levels.
    let min_group_bytes: i64 = if depth < 2 {
        0
    } else {
        (root_retained_bytes / 10000).max(256) as i64
    };

    // Take top N children, collapse the rest
    let max_children = GC_ROOT_MAX_CHILDREN;
    for (i, ((field_key, _class_idx), members)) in groups.iter().enumerate() {
        let group_bytes: i64 = members.iter()
            .map(|&idx| compute_obj_shallow(&objects[idx as usize], oop_size, header_size))
            .sum();
        let group_count = members.len() as u64;

        // Prune tiny groups — collapse instead of expanding
        if group_bytes < min_group_bytes {
            node.collapsed_count += group_count;
            node.collapsed_bytes += group_bytes;
            node.retained_bytes += group_bytes;
            node.retained_count += group_count;
            continue;
        }

        if i < max_children {
            // Expand the first member recursively
            let first_member = members[0];
            let mut child_node = build_single_root_tree(
                first_member, root_idx, objects, root_of, outgoing_refs,
                oop_size, header_size, heap, field_pool, depth + 1,
                root_retained_bytes,
            );
            child_node.field_name = Some(field_pool.get(*field_key).to_string());
            child_node.self_count = group_count;
            child_node.self_bytes = group_bytes;
            // Retained is already computed by recursion for first_member;
            // scale it proportionally for the group
            if child_node.self_count > 0 {
                let scale = group_count as f64 / 1.0_f64;
                child_node.retained_bytes = (child_node.retained_bytes as f64 * scale) as i64;
                child_node.retained_count = (child_node.retained_count as f64 * scale) as u64;
            }
            node.retained_bytes += child_node.retained_bytes;
            node.retained_count += child_node.retained_count;
            node.children.push(child_node);
        } else {
            node.collapsed_count += group_count;
            node.collapsed_bytes += group_bytes;
            node.retained_bytes += group_bytes;
            node.retained_count += group_count;
        }
    }

    node
}
