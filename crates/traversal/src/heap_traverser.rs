use std::collections::hash_map::HashMap;
use std::collections::{HashSet, VecDeque};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use jvm_hprof::{heap_dump, RecordTag};
use memmap2::{Advice, Mmap, MmapMut, UncheckedAdvice};

use ss_heap_model::class_info::JavaClassInfo;
use ss_heap_model::heap_dump::HeapDump;
use ss_heap_model::root::{HeapRootReference, RootType};
use ss_stats::engine::StatsEngine;
use ss_stats::ref_chain::{ChainStep, ClassRefChains, RefChainGroup, RefChainStore};
use ss_stats::traversal_data::{
    entry_of, object_kind, DumpOrder, FieldPool, LiveSet, ObjectEntry, ParentStore, TraversalData,
    NO_PARENT, PARENT_REC_SIZE,
};
use ss_support::external_sort::ExternalSorter;
use ss_support::obj_dir::{ObjDir, ObjRec};

/// BFS object graph traverser. Walks from GC roots, marks visited objects,
/// decodes outgoing references on demand, and accumulates retained sizes per class.
///
/// The traversal keeps no per-object arrays in RAM: object metadata is read from
/// the on-disk object directory, references are decoded from the mapped dump on
/// demand, the BFS parent log is a dense file, and liveness is a bit set.
pub struct HeapObjectTraverser {
    heap: Arc<HeapDump>,
    stats: Arc<StatsEngine>,
}

pub use ss_stats::traversal_data::OutEdge;

/// Record width of the transient (address, dump rank) file handed to the
/// external sorter while building the dump-order table.
const ADDR_RANK_REC: usize = 16;

/// Bytes of record data the sorter may hold at once.
const SORT_MEM_BUDGET: usize = 32 << 20;

/// Object arrays are decoded in slices of at most this many bytes.
const ARRAY_READ_CHUNK: usize = 1 << 20;

/// Rank-based view of the dump: object metadata comes from the on-disk object
/// directory, references come from the mapped dump at the object's own offset.
///
/// This replaces the materialised object table (24 B/object), the sorted address
/// index (4 B/object) and the flat edge table (4 B/object plus 8 B/edge).
struct ObjectStore<'a> {
    heap: &'a HeapDump,
    dir: &'a ObjDir,
    dump: std::fs::File,
    id_bytes: usize,
    class_by_address: &'a HashMap<u64, u32>,
    skip_instance_fields: &'a [(&'a str, &'a str)],
}

impl<'a> ObjectStore<'a> {
    fn len(&self) -> u32 {
        self.dir.len() as u32
    }

    #[inline]
    fn rec(&self, rank: u32) -> Option<ObjRec> {
        self.dir.get(rank as u64).ok()
    }

    #[inline]
    fn entry(&self, rank: u32) -> Option<ObjectEntry> {
        self.rec(rank).map(|r| entry_of(self.heap, r))
    }

    /// Address rank and record of an object address.
    #[inline]
    fn resolve(&self, addr: u64) -> Option<(u32, ObjRec)> {
        self.dir
            .lookup(addr)
            .ok()
            .flatten()
            .map(|(rank, rec)| (rank as u32, rec))
    }

    /// Read len bytes of the dump at offset into buf, returning the number of
    /// bytes actually read (shorter only at end of file).
    fn read_at(&self, offset: u64, len: usize, buf: &mut Vec<u8>) -> usize {
        buf.clear();
        buf.resize(len, 0);
        let mut done = 0usize;
        while done < len {
            match read_some_at(&self.dump, &mut buf[done..], offset + done as u64) {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(_) => break,
            }
        }
        done
    }

    /// The class of a superclass id, through the same address map HPROF parsing uses.
    fn super_class(&self, cid: u64) -> Option<&'a JavaClassInfo> {
        let idx = *self.class_by_address.get(&cid)?;
        self.heap.classes.get(idx as usize)
    }

    /// Sum of the field widths of an instance's class chain.
    fn instance_body_width(&self, cls: &JavaClassInfo) -> usize {
        class_chain_field_bytes(
            cls,
            self.class_by_address,
            &self.heap.classes,
            self.id_bytes,
        )
    }

    /// Decode the outgoing references of one object in HPROF field order,
    /// skipping references that point at class dumps (the BFS never followed
    /// those, and never counted them).
    ///
    /// The callback receives the child address rank, its kind and class index
    /// (both already resolved from the directory) and the field key.
    fn for_each_ref<F>(&self, rank: u32, pool: &mut FieldPool, buf: &mut Vec<u8>, f: &mut F)
    where
        F: FnMut(u32, u8, u32, u32),
    {
        let rec = match self.rec(rank) {
            Some(r) => r,
            None => return,
        };
        match object_kind(rec.kind) {
            0 => {
                let cls = match self.heap.classes.get(rec.class_idx as usize) {
                    Some(c) => c,
                    None => return,
                };
                let need = self.instance_body_width(cls);
                if need == 0 {
                    return;
                }
                let got = self.read_at(rec.file_offset, need, buf);
                if got == 0 {
                    return;
                }
                let mut offset = 0usize;
                let mut cid = Some(cls.class_object_id);
                while let Some(id) = cid {
                    let c = if id == cls.class_object_id {
                        Some(cls)
                    } else {
                        self.super_class(id)
                    };
                    let c = match c {
                        Some(c) => c,
                        None => break,
                    };
                    for fd in &c.instance_fields {
                        if fd.type_code == 2 {
                            let skip = self.skip_instance_fields.iter().any(|(cn, fnn)| {
                                *cn == c.name.as_ref() && *fnn == fd.name.as_ref()
                            });
                            if !skip {
                                if let Some(addr) = read_id_from(&buf[..got], offset, self.id_bytes)
                                {
                                    if addr != 0 {
                                        if let Some((child, crec)) = self.resolve(addr) {
                                            if crec.kind != 3 {
                                                let fk = pool.intern(fd.name.as_ref());
                                                f(child, crec.kind, crec.class_idx, fk);
                                            }
                                        }
                                    }
                                }
                            }
                            offset += self.id_bytes;
                        } else {
                            offset += field_type_size(fd.type_code, self.id_bytes);
                        }
                    }
                    cid = if c.super_class_id != 0 {
                        Some(c.super_class_id)
                    } else {
                        None
                    };
                }
            }
            2 => {
                let len = rec.len as usize;
                if len == 0 || rec.file_offset == 0 {
                    return;
                }
                let mut i = 0usize;
                while i < len {
                    let want = (len - i) * self.id_bytes;
                    let chunk = want.min(ARRAY_READ_CHUNK);
                    let got =
                        self.read_at(rec.file_offset + (i * self.id_bytes) as u64, chunk, buf);
                    if got == 0 {
                        break;
                    }
                    let cnt = got / self.id_bytes;
                    if cnt == 0 {
                        break;
                    }
                    for k in 0..cnt {
                        if let Some(addr) =
                            read_id_from(&buf[..got], k * self.id_bytes, self.id_bytes)
                        {
                            if addr != 0 {
                                if let Some((child, crec)) = self.resolve(addr) {
                                    if crec.kind != 3 {
                                        f(child, crec.kind, crec.class_idx, 0);
                                    }
                                }
                            }
                        }
                    }
                    i += cnt;
                }
            }
            _ => {}
        }
    }
}

fn read_some_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(offset))?;
        f.read(buf)
    }
}

/// Sum of the field widths along an instance's class chain: the number of field
/// bytes HPROF stores for an instance of that class.
fn class_chain_field_bytes(
    cls: &JavaClassInfo,
    class_by_address: &HashMap<u64, u32>,
    classes: &[JavaClassInfo],
    id_bytes: usize,
) -> usize {
    let mut total = 0usize;
    let mut cid = Some(cls.class_object_id);
    while let Some(id) = cid {
        let c = if id == cls.class_object_id {
            Some(cls)
        } else {
            class_by_address
                .get(&id)
                .and_then(|&idx| classes.get(idx as usize))
        };
        match c {
            Some(c) => {
                for fd in &c.instance_fields {
                    total += field_type_size(fd.type_code, id_bytes);
                }
                cid = if c.super_class_id != 0 {
                    Some(c.super_class_id)
                } else {
                    None
                };
            }
            None => break,
        }
    }
    total
}

/// Private scratch directory for the BFS working files: the same one the object
/// directory uses, so the CLI workspace removes it on exit.
fn bfs_temp_dir() -> PathBuf {
    std::env::var_os("SS_MAT_TEMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join(format!("ss-mat-{}", std::process::id())))
}

/// Map the dump for a full sequential pass, keeping its pages out of the way.
struct DumpPass {
    map: Mmap,
    released: std::cell::Cell<usize>,
}

/// File pages a pass releases in one step, and the slack kept behind the cursor.
const RELEASE_STEP: usize = 8 << 20;

impl DumpPass {
    fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        let _ = map.advise(Advice::Sequential);
        Ok(Self {
            map,
            released: std::cell::Cell::new(0),
        })
    }

    fn data(&self) -> &[u8] {
        &self.map[..]
    }

    /// Forget the release cursor, so a second scan of the same file releases its
    /// pages again instead of faulting them all back in.
    fn restart(&self) {
        self.released.set(0);
    }

    /// Release the pages of the part of the file the pass has already consumed.
    ///
    /// Progress is estimated from the fraction of objects visited, and the
    /// estimate holds back 5% of the file so a page is never dropped before it
    /// is read; dropping one early only costs a re-read from the page cache.
    fn progress(&self, done: u64, total: u64) {
        if total == 0 {
            return;
        }
        let len = self.map.len();
        let consumed = ((done as f64 / total as f64) * len as f64 * 0.90) as usize;
        let target = consumed / 4096 * 4096;
        let released = self.released.get();
        if target > released + RELEASE_STEP {
            // SAFETY: the range lies behind the read cursor, so dropping its pages
            // is sound; a later read faults them back in from the file mapping.
            let _ = unsafe {
                self.map
                    .unchecked_advise_range(UncheckedAdvice::DontNeed, released, target - released)
            };
            self.released.set(target);

        }
    }
}

impl Drop for DumpPass {
    fn drop(&mut self) {
        // Release the file pages the pass pulled in instead of leaving them
        // attributed to this process.
        // SAFETY: the pages are only dropped; a later read faults them back in
        // from the read-only file mapping.
        let _ = unsafe { self.map.unchecked_advise(UncheckedAdvice::DontNeed) };
    }
}


impl HeapObjectTraverser {
    pub fn new(heap: Arc<HeapDump>, stats: Arc<StatsEngine>) -> Self {
        Self { heap, stats }
    }

    /// Build the dump-order table, then run BFS straight off the mapped dump.
    pub fn run_traversal(&self, hprof_path: &Path) -> Result<TraversalResult> {
        let dir = self.heap.obj_dir.as_ref().ok_or_else(|| {
            anyhow!("BFS requires the on-disk object directory, which this run does not have")
        })?;
        let object_count = u32::try_from(dir.len())
            .map_err(|_| anyhow!("object directory holds more objects than the BFS can rank"))?;
        let id_bytes = self.heap.identifier_size as usize;

        let oop_size = self.heap.oop_size as usize;
        let header_size = self.heap.header_size as i64;

        // Build class_object_id -> class_idx mapping
        let mut class_obj_to_idx: HashMap<u64, u32> =
            HashMap::with_capacity(self.heap.classes.len());
        let mut class_obj_to_name: HashMap<u64, &str> =
            HashMap::with_capacity(self.heap.classes.len());
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
        let defer_class_names: Vec<&str> = vec![
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
            "java/util/HashMap$Node",
            "java/util/LinkedHashMap$Entry",
            "java/util/LinkedHashMap$HeadTail",
            "java/util/Hashtable$Entry",
            "java/util/concurrent/ConcurrentHashMap$Node",
            "java/util/concurrent/ConcurrentHashMap$ForwardingNode",
            "java/util/concurrent/ConcurrentHashMap$TreeBin",
            "java/util/concurrent/ConcurrentHashMap$TreeNode",
            "java/util/concurrent/ConcurrentHashMap$CounterCell",
            "java/util/TreeMap$Entry",
            "java/util/concurrent/LinkedBlockingQueue$Node",
            "java/util/concurrent/ConcurrentHashMap$ValuesView",
        ];
        let defer_class_indices: HashSet<u32> = defer_class_names
            .iter()
            .filter_map(|name| self.heap.class_by_name.get(*name).copied())
            .collect();

        // Instance fields skipped on every traversal pass. Java only zeroes one.
        let skip_instance_field_pairs: Vec<(&str, &str)> =
            vec![("java/lang/ref/Finalizer", "prev")];

        // Per-class shallow sizes for instances only (kind==0)
        let mut class_inst_sizes: Vec<i64> = vec![0i64; self.heap.classes.len()];

        // Pre-compute field-based aligned body sizes (matching Java's alignedBodySize)
        let ref_sz = self.heap.oop_size as usize;
        let hdr_sz = self.heap.header_size as i64;
        let mut class_aligned_body: Vec<i64> = vec![0i64; self.heap.classes.len()];
        {
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
                    if !visited.insert(cidx) {
                        break;
                    }
                    if cidx >= self.heap.classes.len() {
                        break;
                    }
                    let c = &self.heap.classes[cidx];
                    for fd in &c.instance_fields {
                        total_field_sz += field_type_size(fd.type_code, ref_sz) as i64;
                    }
                    current_id = c.super_class_id;
                }
                let aligned = (hdr_sz + total_field_sz + 7) & !7;
                class_aligned_body[cls.index as usize] = aligned;
            }
        }

        // Field keys the traversal compares against; interning them up front keeps
        // the hot loop on u32 comparisons instead of string comparisons.
        let mut field_pool = FieldPool::new();
        let value_key = field_pool.intern("value");
        let map_key = field_pool.intern("m");
        let backing_map_key = field_pool.intern("backingMap");
        let root_key = field_pool.intern("root");
        let left_key = field_pool.intern("left");
        let right_key = field_pool.intern("right");

        let temp_dir = bfs_temp_dir();
        std::fs::create_dir_all(&temp_dir)?;
        let pid = std::process::id();
        let raw_path = temp_dir.join(format!("bfs-addr-{pid}.tmp"));
        let sorted_path = temp_dir.join(format!("bfs-addr-{pid}.sorted"));
        let order_path = temp_dir.join(format!("bfs-order-{pid}.rec"));
        let parent_path = temp_dir.join(format!("bfs-parent-{pid}.rec"));

        // Phase 1: one sequential pass over the dump records the order the dump
        // lists objects in, plus the class-dump static references (the only
        // class-dump edges that matter, and only for the root tree display).
        eprintln!("BFS Phase 1: reading the dump object order...");
        let mut class_statics: Vec<(u32, Vec<(u64, u32)>)> = Vec::new();
        {
            let pass = DumpPass::open(hprof_path)?;
            let hprof = jvm_hprof::parse_hprof(pass.data())
                .map_err(|e| anyhow!("Failed to re-parse HPROF for BFS: {:?}", e))?;
            let mut out =
                std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&raw_path)?);
            let mut dump_rank: u32 = 0;
            let mut rec = [0u8; ADDR_RANK_REC];
            let mut mismatched_instances: u64 = 0;
            for record in hprof.records_iter() {
                let record = match record {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                if !matches!(record.tag(), RecordTag::HeapDump | RecordTag::HeapDumpSegment) {
                    continue;
                }
                let seg = match record.as_heap_dump_segment() {
                    Some(Ok(seg)) => seg,
                    _ => continue,
                };
                for sr in seg.sub_records() {
                    let addr = match sr {
                        Ok(heap_dump::SubRecord::Instance(inst)) => {
                            let class_idx = class_obj_to_idx
                                .get(&inst.class_obj_id().id())
                                .copied()
                                .unwrap_or(u32::MAX);
                            if let Some(cls) = self.heap.classes.get(class_idx as usize) {
                                let width = class_chain_field_bytes(
                                    cls,
                                    &class_obj_to_idx,
                                    &self.heap.classes,
                                    id_bytes,
                                );
                                if width != inst.fields().len() {
                                    mismatched_instances += 1;
                                }
                            }
                            inst.obj_id().id()
                        }
                        Ok(heap_dump::SubRecord::ObjectArray(arr)) => arr.obj_id().id(),
                        Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => arr.obj_id().id(),
                        Ok(heap_dump::SubRecord::Class(cls)) => {
                            let class_obj_id = cls.obj_id().id();
                            let class_name =
                                class_obj_to_name.get(&class_obj_id).copied().unwrap_or("");
                            let mut refs: Vec<(u64, u32)> = Vec::new();
                            for sf in cls.static_fields() {
                                if let Ok(sf) = sf {
                                    if let heap_dump::FieldValue::ObjectId(Some(id)) = sf.value() {
                                        let addr = id.id();
                                        if addr != 0 {
                                            let field_name_id = sf.name_id().id();
                                            let field_name = self
                                                .heap
                                                .string_table
                                                .get(&field_name_id)
                                                .map(|s| s.as_str())
                                                .unwrap_or("");
                                            let should_skip =
                                                skip_static_pairs.iter().any(|(cn, fn_)| {
                                                    *cn == class_name && *fn_ == field_name
                                                });
                                            if !should_skip {
                                                let fk = if field_name.is_empty() {
                                                    1u32
                                                } else {
                                                    field_pool.intern(field_name)
                                                };
                                                refs.push((addr, fk));
                                            }
                                        }
                                    }
                                }
                            }
                            if !refs.is_empty() {
                                let class_idx = class_obj_to_idx
                                    .get(&class_obj_id)
                                    .copied()
                                    .unwrap_or(u32::MAX);
                                class_statics.push((class_idx, refs));
                            }
                            class_obj_id
                        }
                        _ => continue,
                    };
                    rec[0..8].copy_from_slice(&addr.to_le_bytes());
                    rec[8..12].copy_from_slice(&dump_rank.to_le_bytes());
                    out.write_all(&rec)?;
                    dump_rank += 1;
                    pass.progress(dump_rank as u64, object_count as u64);
                }
            }
            out.flush()?;
            if dump_rank != object_count {
                eprintln!(
                    "BFS: dump lists {} objects, the object directory holds {}",
                    dump_rank, object_count
                );
            }
            if mismatched_instances > 0 {
                eprintln!(
                    "BFS: {} instances carry a field-byte length that differs from their class chain",
                    mismatched_instances
                );
            }
        }

        // Phase 2: sort (address, dump rank) by address and invert it into the
        // dump-order table, so the report can still enumerate objects in the
        // dump's own order while everything else runs on address ranks.
        eprintln!("BFS Phase 2: building the dump-order table...");
        {
            let sorter = ExternalSorter::new(ADDR_RANK_REC, SORT_MEM_BUDGET);
            sorter.sort_file(&raw_path, &sorted_path, |rec| {
                let mut b = [0u8; 8];
                b.copy_from_slice(&rec[0..8]);
                u64::from_le_bytes(b)
            })?;
            let order_file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&order_path)?;
            order_file.set_len(object_count as u64 * 4)?;
            {
                let mut map = unsafe { MmapMut::map_mut(&order_file)? };
                let mut reader =
                    std::io::BufReader::with_capacity(1 << 20, std::fs::File::open(&sorted_path)?);
                let mut rec = [0u8; ADDR_RANK_REC];
                let mut addr_rank: u32 = 0;
                loop {
                    if std::io::Read::read_exact(&mut reader, &mut rec).is_err() {
                        break;
                    }
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&rec[8..12]);
                    let dump_rank = u32::from_le_bytes(b);
                    if (dump_rank as usize) < object_count as usize {
                        let off = dump_rank as usize * 4;
                        map[off..off + 4].copy_from_slice(&addr_rank.to_le_bytes());
                    }
                    addr_rank += 1;
                }
                map.flush()?;
            }
            let _ = std::fs::remove_file(&raw_path);
            let _ = std::fs::remove_file(&sorted_path);
        }
        let order = DumpOrder::open(&order_path, object_count)?;
        ss_support::mem_trace::trace("bfs-object-table");

        // Aggregates that used to fall out of the in-memory object table.
        let mut total_class_objects: u64 = 0;
        for rank in 0..object_count {
            let rec = match dir.get(rank as u64) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if rec.kind == 3 {
                total_class_objects += 1;
            } else if rec.kind == 0 {
                if let Some(sz) = class_aligned_body.get(rec.class_idx as usize) {
                    class_inst_sizes[rec.class_idx as usize] += *sz;
                }
            }
        }

        // Resolve the class-dump static references to address ranks now that the
        // directory can answer address lookups, keeping their dump order.
        let mut class_static_edges: Vec<(u32, Vec<(u32, u32)>)> =
            Vec::with_capacity(class_statics.len());
        let mut statics_slot: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (class_idx, refs) in class_statics.drain(..) {
            let mut resolved = Vec::with_capacity(refs.len());
            for (addr, fk) in refs {
                if let Some((rank, _)) = dir.lookup(addr).ok().flatten() {
                    resolved.push((rank as u32, fk));
                }
            }
            if !resolved.is_empty() {
                statics_slot.insert(class_idx, class_static_edges.len());
                class_static_edges.push((class_idx, resolved));
            }
        }

        let store = ObjectStore {
            heap: &self.heap,
            dir,
            dump: std::fs::File::open(hprof_path)?,
            id_bytes,
            class_by_address: &class_obj_to_idx,
            skip_instance_fields: &skip_instance_field_pairs,
        };

        // Phase 3: BFS traversal from GC roots. Reference lists are decoded from
        // the dump on demand; the parent log goes straight into a mmapped file.
        let mut visited = LiveSet::new(object_count);
        let mut deferred_queue: VecDeque<u32> = VecDeque::new();
        let mut root_of: Vec<u32> = vec![GC_ROOT_GARBAGE_SENTINEL; object_count as usize];
        let mut root_metas: HashMap<u32, ss_stats::ref_chain::GcRootMeta> = HashMap::new();
        let mut queue: VecDeque<u32> = VecDeque::with_capacity(65536);
        let mut ref_buf: Vec<u8> = Vec::with_capacity(4096);

        let parent_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&parent_path)?;
        parent_file.set_len(object_count as u64 * PARENT_REC_SIZE as u64)?;
        let mut parent_map = unsafe { MmapMut::map_mut(&parent_file)? };
        for slot in parent_map.chunks_exact_mut(PARENT_REC_SIZE) {
            slot[0..4].copy_from_slice(&NO_PARENT.to_le_bytes());
            slot[4..8].copy_from_slice(&0u32.to_le_bytes());
        }

        // Java TreeSet wrapper: backing TreeMap entries are live, but their
        // value fields (e.g. TreeSet.PRESENT) are not traversed.
        let tree_set_idx = self.heap.class_by_name.get("java/util/TreeSet").copied();
        let tree_map_idx = self.heap.class_by_name.get("java/util/TreeMap").copied();
        let mut set_backing_maps: HashSet<u32> = HashSet::new();
        let mut set_entries: HashSet<u32> = HashSet::new();

        // Pre-compute which classes are subclasses of java.lang.Thread
        let is_thread_subclass = {
            let mut result = vec![false; self.heap.classes.len()];
            for cls in &self.heap.classes {
                let mut found = cls.name.as_ref() == "java/lang/Thread"
                    || cls.name.as_ref() == "java.lang.Thread";
                let mut current_id = cls.super_class_id;
                let mut visited_set = std::collections::HashSet::new();
                while current_id != 0 && !found {
                    if !visited_set.insert(current_id) {
                        break;
                    }
                    if let Some(&super_idx) = class_obj_to_idx.get(&current_id) {
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
        let classify_root_priority = |root: &HeapRootReference, entry: &ObjectEntry| -> i32 {
            if root.root_type == RootType::SystemClass {
                // Java's filterValidRootRefs drops ALL SYSTEM_CLASS roots because
                // their objectAddress is a class address. Reachability is handled
                // by synthetic JAVA_STATIC roots instead.
                return 0;
            } else {
                let class_idx = entry.class_idx;
                if (class_idx as usize) < self.heap.classes.len() {
                    let cls = &self.heap.classes[class_idx as usize];
                    let name = cls.name.as_ref();
                    // Priority 0: inner class, ThreadGroup, Thread subclass
                    if name.contains(CHR_DOLLAR) {
                        return 0;
                    }
                    if name == "java/lang/ThreadGroup" || name == "java.lang.ThreadGroup" {
                        return 0;
                    }
                    if is_thread_subclass[class_idx as usize] {
                        return 0;
                    }
                    if entry.kind == 0 && defer_class_indices.contains(&class_idx) {
                        return 1;
                    }
                }
            }
            -1 // normal priority
        };

        // Forensic root capture is inert unless SS_MAT_DEBUG_DIR is set.
        let debug_on = ss_stats::debug_dump::enabled();
        let debug_dir = ss_stats::debug_dump::debug_dir();
        let mut root_rows: Vec<crate::debug_dump::RootRow> = if debug_on {
            Vec::with_capacity(self.heap.roots.len())
        } else {
            Vec::new()
        };

        // Seed from GC roots with priority filtering
        for (root_index, root) in self.heap.roots.iter().enumerate() {
            let mut row = if debug_on {
                Some(crate::debug_dump::RootRow {
                    root_index,
                    root_type: root.root_type,
                    java_category: crate::debug_dump::java_root_category(root.root_type),
                    object_id: root.object_id,
                    thread_serial: root.thread_serial,
                    frame_index: root.frame_index,
                    obj_idx: -1,
                    class_name: String::new(),
                    priority: -2,
                    action: "unresolved",
                })
            } else {
                None
            };

            if let Some((rank, rec)) = store.resolve(root.object_id) {
                let entry = entry_of(&self.heap, rec);
                let priority = classify_root_priority(root, &entry);
                if let Some(r) = row.as_mut() {
                    r.obj_idx = rank as i64;
                    r.class_name = get_class_display(entry.class_idx, &self.heap);
                    r.priority = priority;
                }
                if priority != 0 {
                    if !visited.get(rank) {
                        visited.set(rank);
                        root_of[rank as usize] = rank;
                        if priority == 1 {
                            deferred_queue.push_back(rank);
                            if let Some(r) = row.as_mut() {
                                r.action = "seeded_deferred";
                            }
                        } else {
                            queue.push_back(rank);
                            if let Some(r) = row.as_mut() {
                                r.action = "seeded_normal";
                            }
                        }
                    } else if let Some(r) = row.as_mut() {
                        r.action = "already_visited";
                    }
                    use ss_stats::ref_chain::GcRootMeta;
                    root_metas
                        .entry(rank)
                        .and_modify(|m| {
                            if !m.root_types.contains(&root.root_type) {
                                m.root_types.push(root.root_type);
                            }
                        })
                        .or_insert_with(|| GcRootMeta {
                            root_obj_idx: rank,
                            obj_id: root.object_id,
                            class_idx: entry.class_idx,
                            class_name: get_class_display(entry.class_idx, &self.heap),
                            root_types: vec![root.root_type],
                            thread_serial: root.thread_serial,
                        });
                } else if let Some(r) = row.as_mut() {
                    r.action = "dropped_priority0";
                }
            }
            if let Some(r) = row {
                root_rows.push(r);
            }
        }

        let mut visited_count: u64 = 0;
        let mut class_object_visited: u64 = 0;

        // Phase A: Normal BFS
        while let Some(rank) = queue.pop_front() {
            visited_count += 1;
            let rec = match store.rec(rank) {
                Some(r) => r,
                None => continue,
            };
            let entry = entry_of(&self.heap, rec);
            if entry.kind == 3 {
                class_object_visited += 1;
            }

            if (entry.class_idx as usize) < self.heap.classes.len() {
                let shallow = if entry.kind == 0 {
                    class_aligned_body[entry.class_idx as usize]
                } else if entry.kind == 1 {
                    compute_obj_shallow(&entry, oop_size as i32, header_size)
                } else {
                    array_shallow_size(entry.array_length, oop_size as i32, header_size)
                };
                if let Some(pc) = self.stats.get_per_class(entry.class_idx) {
                    pc.retained_size.fetch_add(shallow, Relaxed);
                }
            }

            if entry.kind != 3 {
                prepare_tree_set_entries(
                    &store,
                    rank,
                    entry.class_idx,
                    tree_set_idx,
                    tree_map_idx,
                    &mut field_pool,
                    &mut ref_buf,
                    [map_key, backing_map_key, root_key, left_key, right_key],
                    &mut set_backing_maps,
                    &mut set_entries,
                );
                let in_set_entries = set_entries.contains(&rank);
                store.for_each_ref(rank, &mut field_pool, &mut ref_buf, &mut |child, kind, class_idx, fk| {
                    if in_set_entries && fk == value_key {
                        return;
                    }
                    if visited.get(child) {
                        return;
                    }
                    visited.set(child);
                    let off = child as usize * PARENT_REC_SIZE;
                    parent_map[off..off + 4].copy_from_slice(&rank.to_le_bytes());
                    parent_map[off + 4..off + 8].copy_from_slice(&fk.to_le_bytes());
                    root_of[child as usize] = root_of[rank as usize];
                    if kind == 0 && defer_class_indices.contains(&class_idx) {
                        deferred_queue.push_back(child);
                    } else {
                        queue.push_back(child);
                    }
                });
            }
        }

        // Phase B: Process deferred (weak/finalizer/cleaner) objects. Java's second
        // pass follows every child, so the two queues drain together.
        while !deferred_queue.is_empty() || !queue.is_empty() {
            while let Some(rank) = deferred_queue.pop_front() {
                visited_count += 1;
                let rec = match store.rec(rank) {
                    Some(r) => r,
                    None => continue,
                };
                let entry = entry_of(&self.heap, rec);
                if entry.kind == 3 {
                    class_object_visited += 1;
                }
                if (entry.class_idx as usize) < self.heap.classes.len() {
                    let shallow = if entry.kind == 0 {
                        class_aligned_body[entry.class_idx as usize]
                    } else if entry.kind == 1 {
                        compute_obj_shallow(&entry, oop_size as i32, header_size)
                    } else {
                        array_shallow_size(entry.array_length, oop_size as i32, header_size)
                    };
                    if let Some(pc) = self.stats.get_per_class(entry.class_idx) {
                        pc.retained_size.fetch_add(shallow, Relaxed);
                    }
                }

                prepare_tree_set_entries(
                    &store,
                    rank,
                    entry.class_idx,
                    tree_set_idx,
                    tree_map_idx,
                    &mut field_pool,
                    &mut ref_buf,
                    [map_key, backing_map_key, root_key, left_key, right_key],
                    &mut set_backing_maps,
                    &mut set_entries,
                );
                let in_set_entries = set_entries.contains(&rank);
                store.for_each_ref(rank, &mut field_pool, &mut ref_buf, &mut |child, kind, class_idx, fk| {
                    if in_set_entries && fk == value_key {
                        return;
                    }
                    if visited.get(child) {
                        return;
                    }
                    visited.set(child);
                    let off = child as usize * PARENT_REC_SIZE;
                    parent_map[off..off + 4].copy_from_slice(&rank.to_le_bytes());
                    parent_map[off + 4..off + 8].copy_from_slice(&fk.to_le_bytes());
                    root_of[child as usize] = root_of[rank as usize];
                    if kind == 0 && defer_class_indices.contains(&class_idx) {
                        deferred_queue.push_back(child);
                    } else {
                        queue.push_back(child);
                    }
                });
            }

            // Drain regular queue items discovered from deferred objects
            while let Some(rank) = queue.pop_front() {
                visited_count += 1;
                let rec = match store.rec(rank) {
                    Some(r) => r,
                    None => continue,
                };
                let entry = entry_of(&self.heap, rec);
                if entry.kind == 3 {
                    class_object_visited += 1;
                }
                if (entry.class_idx as usize) < self.heap.classes.len() {
                    let shallow = if entry.kind == 0 {
                        class_aligned_body[entry.class_idx as usize]
                    } else if entry.kind == 1 {
                        compute_obj_shallow(&entry, oop_size as i32, header_size)
                    } else {
                        array_shallow_size(entry.array_length, oop_size as i32, header_size)
                    };
                    if let Some(pc) = self.stats.get_per_class(entry.class_idx) {
                        pc.retained_size.fetch_add(shallow, Relaxed);
                    }
                }

                prepare_tree_set_entries(
                    &store,
                    rank,
                    entry.class_idx,
                    tree_set_idx,
                    tree_map_idx,
                    &mut field_pool,
                    &mut ref_buf,
                    [map_key, backing_map_key, root_key, left_key, right_key],
                    &mut set_backing_maps,
                    &mut set_entries,
                );
                let in_set_entries = set_entries.contains(&rank);
                store.for_each_ref(rank, &mut field_pool, &mut ref_buf, &mut |child, kind, class_idx, fk| {
                    if in_set_entries && fk == value_key {
                        return;
                    }
                    if visited.get(child) {
                        return;
                    }
                    visited.set(child);
                    let off = child as usize * PARENT_REC_SIZE;
                    parent_map[off..off + 4].copy_from_slice(&rank.to_le_bytes());
                    parent_map[off + 4..off + 8].copy_from_slice(&fk.to_le_bytes());
                    root_of[child as usize] = root_of[rank as usize];
                    if kind == 0 && defer_class_indices.contains(&class_idx) {
                        deferred_queue.push_back(child);
                    } else {
                        queue.push_back(child);
                    }
                });
            }
        }

        parent_map.flush()?;
        drop(parent_map);
        drop(parent_file);

        // Count excluding class objects for accurate live/garbage stats
        let total_objects = object_count as u64 - total_class_objects;
        let garbage_count = total_objects - (visited_count - class_object_visited);

        eprintln!(
            "BFS: visited {} / {} objects ({} garbage, {:.1}%), {} class objects excluded",
            visited_count - class_object_visited,
            total_objects,
            garbage_count,
            if total_objects > 0 {
                garbage_count as f64 / total_objects as f64 * 100.0
            } else {
                0.0
            },
            total_class_objects,
        );

        // Update global stats (exclude class objects from counts)
        self.stats.total_instances.store(total_objects, Relaxed);
        self.stats.total_garbage_count.store(garbage_count, Relaxed);

        // Compute total shallow size
        let inst_total: i64 = class_inst_sizes.iter().sum();
        eprintln!(
            "BFS: total instance shallow (field-based) = {} bytes ({} MB)",
            inst_total,
            inst_total / (1024 * 1024)
        );
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
        for rank in 0..object_count {
            let rec = match store.rec(rank) {
                Some(r) => r,
                None => continue,
            };
            let entry = entry_of(&self.heap, rec);
            if visited.get(rank) || entry.kind == 3 {
                continue;
            }
            let shallow = if entry.kind == 0 {
                if (entry.class_idx as usize) < class_aligned_body.len() {
                    class_aligned_body[entry.class_idx as usize]
                } else {
                    ((header_size + entry.body_size as i64 + 7) & !7)
                }
            } else if entry.kind == 1 {
                compute_obj_shallow(&entry, oop_size as i32, header_size)
            } else {
                array_shallow_size(entry.array_length, oop_size as i32, header_size)
            };
            garbage_bytes += shallow;

            if let Some(pc) = self.stats.get_per_class(entry.class_idx) {
                if entry.kind == 0 {
                    pc.instance_garbage_count.fetch_add(1, Relaxed);
                    pc.instance_garbage_bytes.fetch_add(shallow, Relaxed);
                } else if entry.kind == 1 || entry.kind == 2 {
                    pc.array_garbage_count.fetch_add(1, Relaxed);
                    pc.array_garbage_bytes.fetch_add(shallow, Relaxed);
                }
            }
        }
        self.stats.total_garbage_bytes.store(garbage_bytes, Relaxed);

        // Opt-in forensics: dump roots + the full object table right after the
        // BFS has finalised the live set. No effect when the switch is off.
        if let Some(dir) = debug_dir.as_deref() {
            if let Err(e) = crate::debug_dump::write_roots(dir, &root_rows) {
                eprintln!("Debug: failed to write roots.tsv: {}", e);
            }
            if let Err(e) = crate::debug_dump::write_roots_summary(dir, &root_rows) {
                eprintln!("Debug: failed to write roots_summary.txt: {}", e);
            }
            let mut dbg_objects: Vec<ObjectEntry> = Vec::with_capacity(object_count as usize);
            let mut dbg_visited: Vec<bool> = Vec::with_capacity(object_count as usize);
            for rank in 0..object_count {
                if let Some(entry) = store.entry(rank) {
                    dbg_objects.push(entry);
                    dbg_visited.push(visited.get(rank));
                }
            }
            if let Err(e) = crate::debug_dump::write_objects(
                dir,
                &dbg_objects,
                &dbg_visited,
                &self.heap,
                &class_aligned_body,
                visited_count - class_object_visited,
                garbage_count,
            ) {
                eprintln!("Debug: failed to write objects.tsv: {}", e);
            }
            eprintln!(
                "Debug: wrote roots.tsv, roots_summary.txt, objects.tsv to {}",
                dir.display()
            );
        }

        // Phase 3c: Build GC root trees for Section 4
        ss_support::mem_trace::trace("bfs-traversal-done");
        eprintln!("BFS Phase 3c: building GC root trees...");
        let parent = ParentStore::open(&parent_path, object_count)?;
        let mut gc_root_store = build_gc_root_trees(
            &store,
            &root_of,
            &visited,
            &class_static_edges,
            &root_metas,
            oop_size as i32,
            header_size,
            &self.heap,
            &mut field_pool,
            &mut ref_buf,
        );

        // Section 4's garbage row must use the same accounting as Section 5.
        gc_root_store.garbage.count = garbage_count;
        gc_root_store.garbage.bytes = garbage_bytes;
        let mut garbage_top: Vec<(String, u64, i64)> = self
            .stats
            .per_class
            .iter()
            .filter_map(|pc| {
                let count = pc.instance_garbage_count.load(Relaxed) as u64
                    + pc.array_garbage_count.load(Relaxed) as u64;
                let bytes =
                    pc.instance_garbage_bytes.load(Relaxed) + pc.array_garbage_bytes.load(Relaxed);
                if count == 0 {
                    None
                } else {
                    Some((pc.class_name.replace('/', "."), count, bytes))
                }
            })
            .collect();
        garbage_top.sort_by(|a, b| b.2.cmp(&a.2));
        gc_root_store.garbage.top_classes = garbage_top.into_iter().take(20).collect();

        eprintln!(
            "BFS: built {} GC root tree entries",
            gc_root_store.entries.len()
        );
        *self.stats.gc_roots.lock().expect("stats mutex poisoned") = gc_root_store;

        // Phase 4: Reference chain analysis for top classes
        ss_support::mem_trace::trace("bfs-root-trees");
        eprintln!("BFS Phase 4: analyzing reference chains...");
        let ref_chain_store = analyze_ref_chains(
            hprof_path,
            &store,
            &order,
            &parent,
            &mut field_pool,
            &visited,
            &self.heap,
            oop_size,
            header_size,
        );
        ss_support::mem_trace::trace("phase4-ref-chains");
        let chain_count = ref_chain_store.chains.len();

        // Phase 4b: Humongous-specific reference chain analysis for Section 19
        let mut ref_chain_store = ref_chain_store;
        let g1_region_size = self.stats.jvm_info.get().g1_region_size.max(1024 * 1024);
        let humongous_threshold = g1_region_size / 2;
        let humongous_chains = analyze_humongous_chains(
            hprof_path,
            &store,
            &order,
            &parent,
            &mut field_pool,
            &visited,
            &class_static_edges,
            &statics_slot,
            &self.heap,
            oop_size,
            header_size,
            humongous_threshold,
        );
        ss_support::mem_trace::trace("phase4b-humongous");
        ref_chain_store.humongous_chains = humongous_chains;

        *self.stats.ref_chains.lock().expect("stats mutex poisoned") = ref_chain_store;

        // Persist BFS data for post-traversal rendering (ref chains in report).
        *self.stats.traversal_data.lock().expect("stats mutex poisoned") =
            Some(TraversalData {
                heap: Arc::clone(&self.heap),
                object_count,
                order,
                parent,
                root_of,
                visited,
                field_pool,
            });
        eprintln!(
            "BFS: reference chains analyzed for {} classes",
            chain_count
        );

        Ok(TraversalResult {
            visited_count: visited_count - class_object_visited,
            total_objects,
            garbage_count,
            garbage_bytes,
            total_shallow,
        })
    }
}

/// Character constant for the inner-class marker in a JVM class name.
const CHR_DOLLAR: char = '$';

#[allow(clippy::too_many_arguments)]
fn prepare_tree_set_entries(
    store: &ObjectStore<'_>,
    rank: u32,
    class_idx: u32,
    tree_set_idx: Option<u32>,
    tree_map_idx: Option<u32>,
    pool: &mut FieldPool,
    buf: &mut Vec<u8>,
    keys: [u32; 5],
    set_backing_maps: &mut HashSet<u32>,
    set_entries: &mut HashSet<u32>,
) {
    let [map_key, backing_map_key, root_key, left_key, right_key] = keys;
    if Some(class_idx) == tree_set_idx {
        store.for_each_ref(rank, pool, buf, &mut |child, _kind, _cls, fk| {
            if fk == map_key || fk == backing_map_key {
                set_backing_maps.insert(child);
            }
        });
    }
    if Some(class_idx) == tree_map_idx && set_backing_maps.contains(&rank) {
        collect_tree_map_entries(
            store,
            rank,
            pool,
            buf,
            root_key,
            left_key,
            right_key,
            set_entries,
        );
    }
}

/// Collect all red-black-tree entry ranks reachable from a TreeMap's root.
/// Java's TreeSet wrapper traverses the backing TreeMap's entries and enqueues
/// only keys; entry value fields are never followed.
#[allow(clippy::too_many_arguments)]
fn collect_tree_map_entries(
    store: &ObjectStore<'_>,
    map_rank: u32,
    pool: &mut FieldPool,
    buf: &mut Vec<u8>,
    root_key: u32,
    left_key: u32,
    right_key: u32,
    out: &mut HashSet<u32>,
) {
    let mut stack: Vec<u32> = Vec::new();
    store.for_each_ref(map_rank, pool, buf, &mut |child, _kind, _cls, fk| {
        if fk == root_key {
            stack.push(child);
        }
    });
    while let Some(rank) = stack.pop() {
        if !out.insert(rank) {
            continue;
        }
        store.for_each_ref(rank, pool, buf, &mut |child, _kind, _cls, fk| {
            if fk == left_key || fk == right_key {
                stack.push(child);
            }
        });
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

pub(crate) fn array_shallow_size(length: i32, element_size: i32, header_size: i64) -> i64 {
    // Java's ObjectArrayInstance/PrimitiveArrayInstance.getBodySize() uses
    // objectAlignBase = header + Object fields + the 4-byte array length field.
    let data_size = length as i64 * element_size as i64;
    let raw = header_size + 4 + data_size;
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
#[allow(clippy::too_many_arguments)]
fn analyze_ref_chains(
    hprof_path: &Path,
    store: &ObjectStore<'_>,
    order: &DumpOrder,
    parent: &ParentStore,
    field_pool: &mut FieldPool,
    visited: &LiveSet,
    heap: &HeapDump,
    oop_size: usize,
    header_size: i64,
) -> RefChainStore {
    // Find top 50 classes by total_instance_bytes, in one pass over the directory.
    let mut per_class: Vec<i64> = vec![0i64; heap.classes.len()];
    for rank in 0..store.len() {
        let rec = match store.rec(rank) {
            Some(r) => r,
            None => continue,
        };
        if rec.kind == 3 || !visited.get(rank) {
            continue;
        }
        if (rec.class_idx as usize) < per_class.len() {
            let entry = entry_of(heap, rec);
            per_class[rec.class_idx as usize] +=
                compute_obj_shallow(&entry, oop_size as i32, header_size);
        }
    }
    ss_support::mem_trace::trace("chain-class-sizes");
    let mut class_sizes: Vec<(u32, i64)> = per_class
        .into_iter()
        .enumerate()
        .filter(|(_, total)| *total > 0)
        .map(|(idx, total)| (idx as u32, total))
        .collect();
    class_sizes.sort_by(|a, b| b.1.cmp(&a.1));
    let top_classes: Vec<u32> = class_sizes.iter().take(50).map(|(idx, _)| *idx).collect();
    let top_slot: HashMap<u32, usize> = top_classes
        .iter()
        .enumerate()
        .map(|(slot, &cls)| (cls, slot))
        .collect();

    // Collect the sampled object indices of every top class. The old shape walked
    // the object table once per class; one pass over the dump order buckets by
    // class instead, which keeps the per-class insertion order identical.
    let mut live_by_class: Vec<Vec<(i64, u32)>> = vec![Vec::new(); top_classes.len()];
    let mut ref_buf: Vec<u8> = Vec::with_capacity(4096);
    let _ = order.for_each(|_, addr_rank| {
        let rec = match store.rec(addr_rank) {
            Some(r) => r,
            None => return,
        };
        if rec.kind == 3 || !visited.get(addr_rank) {
            return;
        }
        let slot = match top_slot.get(&rec.class_idx) {
            Some(&s) => s,
            None => return,
        };
        let entry = entry_of(heap, rec);
        let mut sz = compute_obj_shallow(&entry, oop_size as i32, header_size);
        // For instance objects (especially String), add the shallow size of
        // referenced arrays (the main data holders) so ranking is meaningful.
        if rec.kind == 0 {
            store.for_each_ref(addr_rank, field_pool, &mut ref_buf, &mut |child, kind, _cls, _fk| {
                if kind == 1 || kind == 2 {
                    if let Some(child_entry) = store.entry(child) {
                        sz += compute_obj_shallow(&child_entry, oop_size as i32, header_size);
                    }
                }
            });
        }
        live_by_class[slot].push((sz, addr_rank));
    });

    let mut all_sampled_indices: Vec<u32> = Vec::new();
    let mut per_class_samples: Vec<(u32, Vec<u32>, Vec<u32>)> = Vec::new();

    for (slot, &class_idx) in top_classes.iter().enumerate() {
        let live_objs = std::mem::take(&mut live_by_class[slot]);
        if live_objs.is_empty() {
            continue;
        }
        let mut live_objs = live_objs;
        // Sort by effective size descending
        live_objs.sort_by(|a, b| b.0.cmp(&a.0));
        let top_by_size: Vec<u32> = live_objs.iter().take(100).map(|(_, idx)| *idx).collect();

        // Random portion: deterministic step sampling from the full set
        let step = (live_objs.len() / 100).max(1);
        let random_samples: Vec<u32> = live_objs
            .iter()
            .step_by(step)
            .take(100)
            .map(|(_, idx)| *idx)
            .collect();

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

    ss_support::mem_trace::trace("chain-samples");
    // Build rich descriptions for all sampled objects via a targeted HPROF pass
    let descriptions = build_sample_descriptions(
        &all_sampled_indices,
        store,
        heap.identifier_size as usize,
        heap,
    );
    ss_support::mem_trace::trace("chain-descriptions");

    let mut store_out = RefChainStore::default();

    for (class_idx, _live_objs, samples) in &per_class_samples {
        let mut groups: HashMap<Vec<(u32, u32)>, Vec<u32>> = HashMap::new();
        for &obj_idx in samples {
            let chain = trace_chain(obj_idx, parent, store);
            groups.entry(chain).or_default().push(obj_idx);
        }

        let mut class_chains = ClassRefChains::default();

        for (path_keys, obj_indices) in &groups {
            let total_bytes: i64 = obj_indices
                .iter()
                .filter_map(|&idx| store.entry(idx))
                .map(|e| compute_obj_shallow(&e, oop_size as i32, header_size))
                .sum();
            let sample_count_in_group = obj_indices.len() as u64;

            // Report the size of the actual sampled group.
            let extrapolated_bytes = total_bytes;
            let extrapolated_count = sample_count_in_group;

            let path: Vec<ChainStep> = path_keys
                .iter()
                .rev()
                .map(|&(cls_idx, fld_key)| {
                    let class_name = if (cls_idx as usize) < heap.classes.len() {
                        heap.classes[cls_idx as usize].name.replace('/', ".")
                    } else {
                        "unknown".to_string()
                    };
                    let field_name = field_pool.get(fld_key).to_string();
                    if field_name == "[elements]" {
                        ChainStep::ArrayElement { class_name }
                    } else if field_name == "[static]" {
                        ChainStep::Root {
                            description: format!("Java Static [{}]", class_name),
                        }
                    } else {
                        ChainStep::Field {
                            class_name,
                            field_name,
                        }
                    }
                })
                .collect();

            let sample_strs: Vec<String> = obj_indices
                .iter()
                .take(8)
                .map(|&idx| {
                    descriptions
                        .get(&idx)
                        .cloned()
                        .unwrap_or_else(|| describe_object_simple(idx, store, heap))
                })
                .collect();
            let sample_obj_ids: Vec<u64> = obj_indices
                .iter()
                .take(8)
                .map(|&idx| store.rec(idx).map(|r| r.addr).unwrap_or(0))
                .collect();

            let group = RefChainGroup {
                size_bytes: extrapolated_bytes,
                count: extrapolated_count,
                samples: sample_strs,
                sample_obj_ids,
                path,
            };

            if path_keys.len() <= 2 {
                class_chains.expensive_fields.push(group);
            } else {
                class_chains.full_chains.push(group);
            }
        }

        class_chains
            .expensive_fields
            .sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        class_chains
            .full_chains
            .sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
        class_chains.expensive_fields.truncate(10);
        class_chains.full_chains.truncate(10);

        store_out.chains.insert(*class_idx, class_chains);
    }

    store_out
}

/// Analyze reference chains specifically for humongous objects (>= half G1 region size).
#[allow(clippy::too_many_arguments)]
fn analyze_humongous_chains(
    hprof_path: &Path,
    store: &ObjectStore<'_>,
    order: &DumpOrder,
    parent: &ParentStore,
    field_pool: &mut FieldPool,
    visited: &LiveSet,
    class_static_edges: &[(u32, Vec<(u32, u32)>)],
    statics_slot: &HashMap<u32, usize>,
    heap: &HeapDump,
    oop_size: usize,
    header_size: i64,
    humongous_threshold: i64,
) -> ClassRefChains {
    // Collect all humongous objects (both live and garbage)
    let mut humongous_objs: Vec<u32> = Vec::new();
    for rank in 0..store.len() {
        let rec = match store.rec(rank) {
            Some(r) => r,
            None => continue,
        };
        if rec.kind == 3 {
            continue;
        }
        let entry = entry_of(heap, rec);
        let size = compute_obj_shallow(&entry, oop_size as i32, header_size);
        if size >= humongous_threshold {
            humongous_objs.push(rank);
        }
    }

    if humongous_objs.is_empty() {
        return ClassRefChains::default();
    }

    // Build rich descriptions for humongous objects
    let descriptions =
        build_sample_descriptions(&humongous_objs, store, heap.identifier_size as usize, heap);

    // Chains of unreachable humongous objects need their first incoming edge, so
    // a reverse index is built in the dump's own order. Only humongous targets are
    // remembered: one entry each instead of two arrays sized by the object count.
    // Chains of unreachable humongous objects are rebuilt from their first
    // incoming edge, and the walk continues through ordinary objects, so the
    // reverse index covers every object. It lives in a dense 8-byte-per-object
    // file instead of two arrays sized by the object count.
    let first_src = FirstSrc::build(store, order, field_pool, class_static_edges, statics_slot)
        .unwrap_or_else(|| FirstSrc::empty());

    // Trace chains using the BFS parent for live objects, the reverse ref otherwise
    let mut groups: HashMap<Vec<(u32, u32)>, Vec<u32>> = HashMap::new();
    for &obj_idx in &humongous_objs {
        let chain = trace_humongous_chain(obj_idx, parent, store, &first_src);
        groups.entry(chain).or_default().push(obj_idx);
    }

    let mut class_chains = ClassRefChains::default();

    for (path_keys, obj_indices) in &groups {
        let total_bytes: i64 = obj_indices
            .iter()
            .filter_map(|&idx| store.entry(idx))
            .map(|e| compute_obj_shallow(&e, oop_size as i32, header_size))
            .sum();
        let count = obj_indices.len() as u64;

        let path: Vec<ChainStep> = path_keys
            .iter()
            .map(|&(cls_idx, fld_key)| {
                if cls_idx == u32::MAX {
                    return ChainStep::Root {
                        description: "Unreachable".to_string(),
                    };
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
                    ChainStep::Root {
                        description: format!("Java Static [{}]", class_name),
                    }
                } else {
                    ChainStep::Field {
                        class_name,
                        field_name,
                    }
                }
            })
            .collect();

        let sample_strs: Vec<String> = obj_indices
            .iter()
            .take(8)
            .map(|&idx| {
                descriptions
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(|| describe_object_simple(idx, store, heap))
            })
            .collect();
        let sample_obj_ids: Vec<u64> = obj_indices
            .iter()
            .take(8)
            .map(|&idx| store.rec(idx).map(|r| r.addr).unwrap_or(0))
            .collect();

        let group = RefChainGroup {
            size_bytes: total_bytes,
            count,
            samples: sample_strs,
            sample_obj_ids,
            path: path.clone(),
        };

        class_chains.expensive_fields.push(group.clone());
        class_chains.full_chains.push(group);
    }

    class_chains
        .expensive_fields
        .sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    class_chains
        .full_chains
        .sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

    class_chains
}

/// Build rich sample descriptions for the sampled objects.
///
/// A description needs the object's body, not the dump: the object directory
/// already knows where every body sits, so the samples are read by offset and the
/// dump is never mapped here. Re-parsing even one segment of a large dump would
/// pull its pages back into the resident set for the sake of a few thousand
/// samples. Each body is copied into one small synthetic heap-dump record and
/// decoded by the same reader the full parse uses, so the field semantics stay
/// exactly the ones the full parse would have given.
fn build_sample_descriptions(
    sampled_indices: &[u32],
    store: &ObjectStore<'_>,
    id_bytes: usize,
    heap: &HeapDump,
) -> HashMap<u32, String> {
    let mut sampled_ids: HashMap<u64, u32> = HashMap::with_capacity(sampled_indices.len());
    for &idx in sampled_indices {
        if let Some(entry) = store.entry(idx) {
            sampled_ids.insert(entry.obj_id, idx);
        }
    }

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

    let mut results: HashMap<u32, String> = HashMap::new();
    let mut bytearray_content: HashMap<u64, Vec<u8>> = HashMap::new();
    let mut string_values: Vec<(u32, u64)> = Vec::new();

    // Objects whose whole body is small enough to be cached for String resolution.
    let mut cache_content: HashSet<u64> = HashSet::new();
    let mut queued: HashSet<u64> = HashSet::new();
    // Sub-record under construction, plus the buffer it is decoded from. Both are
    // reused, so a large sampled array costs one copy instead of one per sample.
    let mut body: Vec<u8> = Vec::new();
    let mut mini: Vec<u8> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();

    // Pass 1: the sampled String instances. Their value arrays must be known before
    // the arrays are read, because the String text comes out of the array body.
    let mut string_value_targets: HashMap<u64, u32> = HashMap::new(); // array_obj_id -> string_idx
    for &idx in sampled_indices {
        let rec = match store.rec(idx) {
            Some(r) => r,
            None => continue,
        };
        if rec.kind != 0 || rec.file_offset == 0 || string_class_idx != Some(rec.class_idx) {
            continue;
        }
        let (offset, field_size) = match string_value_offset {
            Some(v) => v,
            None => continue,
        };
        let cls = match heap.classes.get(rec.class_idx as usize) {
            Some(c) => c,
            None => continue,
        };
        // The value reference sits inside the instance body at the offset the
        // class chain gives; that prefix is all the description reads.
        let width = offset + field_size;
        let got = store.read_at(rec.file_offset, width, &mut buf);
        if got < width {
            continue;
        }
        let arr_id = match read_id_from(&buf[..got], offset, field_size) {
            Some(id) => id,
            None => continue,
        };
        if !queued.insert(rec.addr) {
            continue;
        }
        string_value_targets.insert(arr_id, idx);
        body.clear();
        push_instance_record(&mut body, rec.addr, cls.class_object_id, &buf[..got], id_bytes);
        decode_sample_record(
            &mut mini,
            &body,
            id_bytes,
            heap,
            store,
            &sampled_ids,
            &string_value_targets,
            string_class_idx,
            string_value_offset,
            &cache_content,
            &mut results,
            &mut bytearray_content,
            &mut string_values,
        );
    }

    // Pass 2: every sampled array, plus the value arrays the Strings above point at.
    for &idx in sampled_indices {
        let rec = match store.rec(idx) {
            Some(r) => r,
            None => continue,
        };
        body.clear();
        if !queue_array_record(&mut body, &mut cache_content, &mut queued, &mut buf, store, rec) {
            continue;
        }
        decode_sample_record(
            &mut mini,
            &body,
            id_bytes,
            heap,
            store,
            &sampled_ids,
            &string_value_targets,
            string_class_idx,
            string_value_offset,
            &cache_content,
            &mut results,
            &mut bytearray_content,
            &mut string_values,
        );
    }
    let mut value_arrays: Vec<u64> = string_value_targets.keys().copied().collect();
    value_arrays.sort_unstable();
    for arr_id in value_arrays {
        let rec = match store.resolve(arr_id) {
            Some((_, r)) => r,
            None => continue,
        };
        body.clear();
        if !queue_array_record(&mut body, &mut cache_content, &mut queued, &mut buf, store, rec) {
            continue;
        }
        decode_sample_record(
            &mut mini,
            &body,
            id_bytes,
            heap,
            store,
            &sampled_ids,
            &string_value_targets,
            string_class_idx,
            string_value_offset,
            &cache_content,
            &mut results,
            &mut bytearray_content,
            &mut string_values,
        );
    }

    // Resolve String content using collected byte[] data
    for (string_idx, value_array_id) in string_values {
        if let Some(bytes) = bytearray_content.get(&value_array_id) {
            let text = String::from_utf8_lossy(bytes);
            let truncated: String = text.chars().take(200).collect();
            if !truncated.is_empty() {
                results.insert(
                    string_idx,
                    format!("<t-c>String</t-c> \"{}\"", escape_html(&truncated)),
                );
            }
        }
    }

    // Fill in defaults for any sampled objects not yet described
    for &idx in sampled_indices {
        if !results.contains_key(&idx) {
            results.insert(idx, describe_object_simple(idx, store, heap));
        }
    }

    results
}

/// HPROF header the synthetic sample record is decoded under; the version label is
/// never inspected by the reader, only the identifier size that follows it.
const SYNTH_HEADER: &[u8] = b"JAVA PROFILE 1.0.2\0";

/// Top-level HPROF tag of a heap-dump segment record.
const SR_HEAP_DUMP_SEGMENT: u8 = 0x1C;

/// Element counts the sample descriptions render: primitive arrays show the first
/// 65,536 elements, object arrays the first five. Longer bodies are never copied.
const PRIM_SAMPLE_ELEMENTS: usize = 65536;
const OBJ_ARRAY_SAMPLE_ELEMENTS: usize = 5;

fn push_id(out: &mut Vec<u8>, value: u64, id_bytes: usize) {
    if id_bytes == 4 {
        out.extend_from_slice(&(value as u32).to_be_bytes());
    } else {
        out.extend_from_slice(&value.to_be_bytes());
    }
}

/// Append one instance sub-record carrying the body read from the dump.
fn push_instance_record(
    out: &mut Vec<u8>,
    addr: u64,
    class_obj_id: u64,
    fields: &[u8],
    id_bytes: usize,
) {
    out.push(0x21);
    push_id(out, addr, id_bytes);
    out.extend_from_slice(&0u32.to_be_bytes());
    push_id(out, class_obj_id, id_bytes);
    out.extend_from_slice(&(fields.len() as u32).to_be_bytes());
    out.extend_from_slice(fields);
}

/// Copy the part of an array body its description renders into the sample record,
/// returning false when the object is not an array the descriptions cover.
///
/// Only a primitive array whose whole body is short enough to be cached for String
/// resolution is marked in cache_content; a longer one contributes its rendered
/// prefix alone, exactly as the full-dump walk did.
fn queue_array_record(
    body: &mut Vec<u8>,
    cache_content: &mut HashSet<u64>,
    queued: &mut HashSet<u64>,
    buf: &mut Vec<u8>,
    store: &ObjectStore<'_>,
    rec: ObjRec,
) -> bool {
    // Directory kind 2 is a primitive array, kind 1 an object array.
    if rec.file_offset == 0 || (rec.kind != 1 && rec.kind != 2) {
        return false;
    }
    if !queued.insert(rec.addr) {
        return false;
    }
    let id_bytes = store.id_bytes;
    let len = rec.len as usize;
    if rec.kind == 2 {
        let width = rec.elem as usize;
        if width == 0 {
            return false;
        }
        let want = len.min(PRIM_SAMPLE_ELEMENTS);
        let got = store.read_at(rec.file_offset, want * width, buf);
        let count = got / width;
        body.push(0x23);
        push_id(body, rec.addr, id_bytes);
        body.extend_from_slice(&0u32.to_be_bytes());
        body.extend_from_slice(&(count as u32).to_be_bytes());
        body.push(rec.flags as u8);
        body.extend_from_slice(&buf[..count * width]);
        if count == len && len <= PRIM_SAMPLE_ELEMENTS {
            cache_content.insert(rec.addr);
        }
    } else {
        let want = len.min(OBJ_ARRAY_SAMPLE_ELEMENTS);
        let got = store.read_at(rec.file_offset, want * id_bytes, buf);
        let count = got / id_bytes;
        body.push(0x22);
        push_id(body, rec.addr, id_bytes);
        body.extend_from_slice(&0u32.to_be_bytes());
        body.extend_from_slice(&(count as u32).to_be_bytes());
        // Element class is not read back: the display name comes from the directory.
        push_id(body, 0, id_bytes);
        body.extend_from_slice(&buf[..count * id_bytes]);
    }
    true
}

/// Decode one copied sample sub-record and fold what it carries into the results.
#[allow(clippy::too_many_arguments)]
fn decode_sample_record(
    mini: &mut Vec<u8>,
    body: &[u8],
    id_bytes: usize,
    heap: &HeapDump,
    store: &ObjectStore<'_>,
    sampled_ids: &HashMap<u64, u32>,
    string_value_targets: &HashMap<u64, u32>,
    string_class_idx: Option<u32>,
    string_value_offset: Option<(usize, usize)>,
    cache_content: &HashSet<u64>,
    results: &mut HashMap<u32, String>,
    bytearray_content: &mut HashMap<u64, Vec<u8>>,
    string_values: &mut Vec<(u32, u64)>,
) {
    if body.is_empty() {
        return;
    }
    mini.clear();
    mini.extend_from_slice(SYNTH_HEADER);
    mini.extend_from_slice(&(id_bytes as u32).to_be_bytes());
    mini.extend_from_slice(&0u64.to_be_bytes());
    mini.push(SR_HEAP_DUMP_SEGMENT);
    mini.extend_from_slice(&0u32.to_be_bytes());
    mini.extend_from_slice(&(body.len() as u32).to_be_bytes());
    mini.extend_from_slice(body);

    let hprof = match jvm_hprof::parse_hprof(mini) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "Warning: could not decode a sampled object body ({:?}); it falls back to its identity",
                e
            );
            return;
        }
    };
    let id_size = hprof.header().id_size();
    let mut records = hprof.records_iter();
    while let Some(Ok(record)) = records.next() {
        let seg = match record.as_heap_dump_segment() {
            Some(Ok(seg)) => seg,
            _ => continue,
        };
        for sr in seg.sub_records() {
            match sr {
                Ok(heap_dump::SubRecord::PrimitiveArray(arr)) => {
                    let obj_id = arr.obj_id().id();
                    let is_sampled = sampled_ids.contains_key(&obj_id);
                    let is_string_value = string_value_targets.contains_key(&obj_id);

                    if is_sampled || is_string_value {
                        // Cache byte content for String resolution (limit size)
                        if cache_content.contains(&obj_id) {
                            bytearray_content.insert(obj_id, collect_prim_bytes(&arr));
                        }

                        if let Some(&idx) = sampled_ids.get(&obj_id) {
                            let entry = match store.entry(idx) {
                                Some(e) => e,
                                None => continue,
                            };
                            let base_type = arr.primitive_type().java_type_name();
                            let desc =
                                format_prim_array_sample(&arr, base_type, entry.array_length);
                            results.insert(idx, desc);
                        }
                    }
                }
                Ok(heap_dump::SubRecord::ObjectArray(arr)) => {
                    let obj_id = arr.obj_id().id();
                    if let Some(&idx) = sampled_ids.get(&obj_id) {
                        let entry = match store.entry(idx) {
                            Some(e) => e,
                            None => continue,
                        };
                        let class_name = get_class_display(entry.class_idx, heap);
                        let array_length = entry.array_length;
                        let mut elem_descs: Vec<String> = Vec::new();
                        for elem in arr.elements(id_size).take(5) {
                            if let Some(id) = elem.ok().flatten() {
                                let addr = id.id();
                                if addr != 0 {
                                    if let Some(elem_class_idx) = heap.class_idx_of(addr) {
                                        let elem_name = get_class_display(elem_class_idx, heap);
                                        if elem_name == "java.lang.String" {
                                            elem_descs
                                                .push(format!("<t-c>String</t-c>@{:x}", addr));
                                        } else {
                                            elem_descs.push(format!("<t-c>{}</t-c>", elem_name));
                                        }
                                    }
                                }
                            }
                        }
                        if elem_descs.is_empty() {
                            results.insert(
                                idx,
                                format!(
                                    "<t-c>{}</t-c>[{}]",
                                    class_name,
                                    format_number(array_length as i64)
                                ),
                            );
                        } else {
                            let trailing = if array_length > 5 {
                                format!(", ... ({} more)", array_length - 5)
                            } else {
                                String::new()
                            };
                            results.insert(
                                idx,
                                format!(
                                    "<t-c>{}</t-c>[{}]{{<t-s>{}{}</t-s>}}",
                                    class_name,
                                    format_number(array_length as i64),
                                    elem_descs.join(", "),
                                    trailing
                                ),
                            );
                        }
                    }
                }
                Ok(heap_dump::SubRecord::Instance(inst)) => {
                    let obj_id = inst.obj_id().id();
                    if let Some(&idx) = sampled_ids.get(&obj_id) {
                        let entry = match store.entry(idx) {
                            Some(e) => e,
                            None => continue,
                        };
                        if string_class_idx == Some(entry.class_idx) {
                            if let Some((offset, field_size)) = string_value_offset {
                                if let Some(arr_id) =
                                    read_id_from(inst.fields(), offset, field_size)
                                {
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
/// First incoming edge of every object, in a dense 8-byte-per-object file.
///
/// Only objects that never became reachable need it, but the walk continues
/// through ordinary objects, so the index is dense. The file goes away with the
/// store.
struct FirstSrc {
    file: std::fs::File,
    path: PathBuf,
    len: u32,
}

impl FirstSrc {
    fn build(
        store: &ObjectStore<'_>,
        order: &DumpOrder,
        field_pool: &mut FieldPool,
        class_static_edges: &[(u32, Vec<(u32, u32)>)],
        statics_slot: &HashMap<u32, usize>,
    ) -> Option<Self> {
        let len = store.len();
        let path = bfs_temp_dir().join(format!("bfs-firstsrc-{}.rec", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .ok()?;
        file.set_len(len as u64 * PARENT_REC_SIZE as u64).ok()?;
        let mut ref_buf: Vec<u8> = Vec::with_capacity(4096);
        let mut pending: Vec<(u32, u32)> = Vec::new();
        {
            let mut map = unsafe { MmapMut::map_mut(&file).ok()? };
            for slot in map.chunks_exact_mut(PARENT_REC_SIZE) {
                slot[0..4].copy_from_slice(&NO_PARENT.to_le_bytes());
                slot[4..8].copy_from_slice(&0u32.to_le_bytes());
            }
            let _ = order.for_each(|_, addr_rank| {
                let rec = match store.rec(addr_rank) {
                    Some(r) => r,
                    None => return,
                };
                if rec.kind == 3 {
                    // A class dump contributes its static field references; the
                    // dump lists class dumps first, so they are the first
                    // referrer of anything only a static field points at.
                    if let Some(&idx) = statics_slot.get(&rec.class_idx) {
                        for &(target, fk) in &class_static_edges[idx].1 {
                            let off = target as usize * PARENT_REC_SIZE;
                            if map[off..off + 4] == NO_PARENT.to_le_bytes() {
                                map[off..off + 4].copy_from_slice(&addr_rank.to_le_bytes());
                                map[off + 4..off + 8].copy_from_slice(&fk.to_le_bytes());
                            }
                        }
                    }
                    return;
                }
                pending.clear();
                store.for_each_ref(addr_rank, field_pool, &mut ref_buf, &mut |child, _k, _c, fk| {
                    pending.push((child, fk));
                });
                for &(child, fk) in pending.iter() {
                    let off = child as usize * PARENT_REC_SIZE;
                    if map[off..off + 4] == NO_PARENT.to_le_bytes() {
                        map[off..off + 4].copy_from_slice(&addr_rank.to_le_bytes());
                        map[off + 4..off + 8].copy_from_slice(&fk.to_le_bytes());
                    }
                }
            });
            map.flush().ok()?;
        }
        Some(Self { file, path, len })
    }

    /// Degenerate store used when the index file cannot be created.
    fn empty() -> Self {
        Self {
            file: std::fs::File::open("/dev/null").expect("/dev/null"),
            path: PathBuf::new(),
            len: 0,
        }
    }

    fn get(&self, rank: u32) -> Option<(u32, u32)> {
        if rank >= self.len {
            return None;
        }
        let mut buf = [0u8; PARENT_REC_SIZE];
        read_some_at(&self.file, &mut buf, rank as u64 * PARENT_REC_SIZE as u64).ok()?;
        let mut p = [0u8; 4];
        p.copy_from_slice(&buf[0..4]);
        let src = u32::from_le_bytes(p);
        if src == NO_PARENT {
            return None;
        }
        let mut k = [0u8; 4];
        k.copy_from_slice(&buf[4..8]);
        Some((src, u32::from_le_bytes(k)))
    }
}

impl Drop for FirstSrc {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Trace from an object up to root via the parent log, returning
/// (parent class index, field key) pairs.
fn trace_chain(start: u32, parent: &ParentStore, store: &ObjectStore<'_>) -> Vec<(u32, u32)> {
    let mut chain = Vec::new();
    let mut current = start;
    for _ in 0..60 {
        let (p, edge_key) = parent.parent_of(current);
        if p == NO_PARENT {
            break;
        }
        let parent_class = store.rec(p).map(|r| r.class_idx).unwrap_or(u32::MAX);
        chain.push((parent_class, edge_key));
        current = p;
    }
    chain
}

/// Trace chain for humongous objects, handling both visited (BFS parent) and
/// garbage (reverse ref lookup) objects. Returns (parent class index, field key)
/// pairs from the humongous object upward. If the chain reaches a garbage object
/// without further parents, appends a "virtual root" marker.
fn trace_humongous_chain(
    start: u32,
    parent: &ParentStore,
    store: &ObjectStore<'_>,
    first_src: &FirstSrc,
) -> Vec<(u32, u32)> {
    let mut chain = Vec::new();
    let mut current = start;

    for _ in 0..60 {
        let (p, edge_key) = parent.parent_of(current);
        if p != NO_PARENT {
            // Has BFS parent — normal case
            let parent_class = store.rec(p).map(|r| r.class_idx).unwrap_or(u32::MAX);
            chain.push((parent_class, edge_key));
            current = p;
        } else {
            // No BFS parent — try the first referrer (garbage object)
            match first_src.get(current) {
                Some((src_idx, field_key)) => {
                    let src_class = store.rec(src_idx).map(|r| r.class_idx).unwrap_or(u32::MAX);
                    chain.push((src_class, field_key));
                    current = src_idx;
                }
                None => {
                    // No parent and no reverse ref — truly unreachable
                    chain.push((u32::MAX, u32::MAX));
                    break;
                }
            }
        }
    }
    chain
}

/// Simple object description fallback (no content).
fn describe_object_simple(obj_idx: u32, store: &ObjectStore<'_>, heap: &HeapDump) -> String {
    let entry = match store.entry(obj_idx) {
        Some(e) => e,
        None => return String::new(),
    };
    let class_name = get_class_display(entry.class_idx, heap);
    match entry.kind {
        1 | 2 => format!(
            "<t-c>{}</t-c>[{}]",
            class_name,
            format_number(entry.array_length as i64)
        ),
        _ => format!("<t-c>{}</t-c>", class_name),
    }
}

/// Compute shallow size of an object entry.
pub(crate) fn compute_obj_shallow(obj: &ObjectEntry, oop_size: i32, header_size: i64) -> i64 {
    match obj.kind {
        0 => (header_size + obj.body_size as i64 + 7) & !7,
        1 => array_shallow_size(obj.array_length, obj.element_size as i32, header_size),
        2 => array_shallow_size(obj.array_length, oop_size, header_size),
        _ => 0,
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

const GC_ROOT_MAX_DEPTH: usize = 10;
const GC_ROOT_MAX_CHILDREN: usize = 12;
const GC_ROOT_TOP_N: usize = 100;
const GC_ROOT_SAMPLE_N: usize = 40;

#[allow(clippy::too_many_arguments)]
fn build_gc_root_trees(
    store: &ObjectStore<'_>,
    root_of: &[u32],
    visited: &LiveSet,
    class_static_edges: &[(u32, Vec<(u32, u32)>)],
    root_metas: &HashMap<u32, ss_stats::ref_chain::GcRootMeta>,
    oop_size: i32,
    header_size: i64,
    heap: &HeapDump,
    field_pool: &mut FieldPool,
    buf: &mut Vec<u8>,
) -> ss_stats::ref_chain::GcRootStore {
    // Step 1: Aggregate per-root retained sizes (keyed by root rank)
    let mut root_retained: HashMap<u32, (i64, u64)> = HashMap::new();
    let mut garbage_bytes: i64 = 0;
    let mut garbage_count: u64 = 0;

    for rank in 0..store.len() {
        let rec = match store.rec(rank) {
            Some(r) => r,
            None => continue,
        };
        if rec.kind == 3 {
            continue;
        }
        let entry = entry_of(heap, rec);
        let r = root_of[rank as usize];
        if r == GC_ROOT_GARBAGE_SENTINEL {
            if !visited.get(rank) {
                garbage_bytes += compute_obj_shallow(&entry, oop_size, header_size);
                garbage_count += 1;
            }
            continue;
        }
        let shallow = compute_obj_shallow(&entry, oop_size, header_size);
        let e = root_retained.entry(r).or_insert((0, 0));
        e.0 += shallow;
        e.1 += 1;
    }

    // Step 1b: Collect "display-only" JAVA_STATIC roots from class objects'
    // static field references. They only add Section 4 display entries labelled
    // "Java Static [ClassName.field]" (matching Java's JAVA_STATIC roots).
    let mut static_display_roots: Vec<(u32, String, u32)> = Vec::new(); // (rank, label, class_idx)
    for (class_idx, edges) in class_static_edges {
        for &(target, field_key) in edges {
            if target >= visited.len() {
                continue;
            }
            if !visited.get(target) {
                continue;
            }
            let trec = match store.rec(target) {
                Some(r) => r,
                None => continue,
            };
            if trec.kind == 3 {
                continue;
            }
            // Only add as a static display root if the target is NOT already a
            // real root.
            if root_of[target as usize] == target {
                continue;
            }
            let owning_class_name = if (*class_idx as usize) < heap.classes.len() {
                heap.classes[*class_idx as usize].name.replace('/', ".")
            } else {
                String::new()
            };
            let field_name = field_pool.get(field_key).to_string();
            let label = format!("Java Static [{}.{}]", owning_class_name, field_name);
            static_display_roots.push((target, label, trec.class_idx));
        }
    }

    // Step 2: Group roots by (primary_root_type, class_idx)
    let mut groups: HashMap<(RootType, u32), Vec<(u32, i64, u64)>> = HashMap::new();
    for (root_idx, retained_bytes, retained_count) in
        root_retained.iter().map(|(&k, &(b, c))| (k, b, c))
    {
        let meta = match root_metas.get(&root_idx) {
            Some(m) => m,
            None => continue,
        };
        let primary_type = meta.root_types.iter().find(|&&t| t != RootType::Unknown).copied();
        let primary_type = match primary_type {
            Some(t) => t,
            None => continue,
        };
        if primary_type == RootType::JavaStackFrame {
            continue;
        }
        let key = (primary_type, meta.class_idx);
        groups.entry(key).or_default().push((root_idx, retained_bytes, retained_count));
    }

    // Step 2b: Add display-only JAVA_STATIC roots as separate entries.
    for &(obj_idx, _, _) in &static_display_roots {
        let synthetic_key = 0x8000_0000 | obj_idx;
        groups.insert(
            (RootType::SystemClass, synthetic_key),
            vec![(obj_idx, 0, 0)],
        );
    }

    // Step 3: For each group, sum retained bytes and pick the largest member.
    let mut sorted_groups: Vec<(RootType, u32, u32, i64, u64)> = groups
        .into_iter()
        .map(|((rt, class_idx), members)| {
            let total_bytes: i64 = members.iter().map(|(_, b, _)| *b).sum();
            let total_count: u64 = members.iter().map(|(_, _, c)| *c).sum();
            let largest = members
                .iter()
                .max_by_key(|(_, b, _)| *b)
                .map(|(idx, _, _)| *idx)
                .unwrap_or(0);
            (rt, class_idx, largest, total_bytes, total_count)
        })
        .collect();
    sorted_groups.sort_by(|a, b| b.3.cmp(&a.3));

    // Step 4: Build trees for top N groups
    let top_n = GC_ROOT_TOP_N.min(sorted_groups.len());
    let mut entries: Vec<ss_stats::ref_chain::GcRootEntry> = Vec::with_capacity(top_n);

    for &(rt, class_key, root_idx, retained_bytes, retained_count) in &sorted_groups[..top_n] {
        let is_static_display = class_key & 0x8000_0000 != 0;
        let meta = if is_static_display {
            let label = static_display_roots
                .iter()
                .find(|&&(idx, _, _)| idx == root_idx)
                .map(|(_, l, _)| l.clone())
                .unwrap_or_else(|| "Java Static".to_string());
            let entry = store.entry(root_idx);
            ss_stats::ref_chain::GcRootMeta {
                root_obj_idx: root_idx,
                obj_id: entry.map(|e| e.obj_id).unwrap_or(0),
                class_idx: entry.map(|e| e.class_idx).unwrap_or(u32::MAX),
                class_name: label,
                root_types: vec![RootType::SystemClass],
                thread_serial: None,
            }
        } else {
            root_metas.get(&root_idx).cloned().unwrap_or_else(|| {
                let class_idx = store.rec(root_idx).map(|r| r.class_idx).unwrap_or(u32::MAX);
                ss_stats::ref_chain::GcRootMeta {
                    root_obj_idx: root_idx,
                    obj_id: store.rec(root_idx).map(|r| r.addr).unwrap_or(0),
                    class_idx,
                    class_name: get_class_display(class_idx, heap),
                    root_types: vec![RootType::Unknown],
                    thread_serial: None,
                }
            })
        };

        let self_bytes = store
            .entry(root_idx)
            .map(|e| compute_obj_shallow(&e, oop_size, header_size))
            .unwrap_or(0);

        let tree = build_single_root_tree(
            root_idx,
            root_idx,
            store,
            root_of,
            field_pool,
            buf,
            oop_size,
            header_size,
            heap,
            0,
            retained_bytes.max(self_bytes),
        );

        entries.push(ss_stats::ref_chain::GcRootEntry {
            meta,
            retained_bytes: retained_bytes.max(self_bytes),
            retained_count: retained_count.max(1),
            self_bytes,
            tree,
        });
    }

    // Step 4b: Compute remaining groups summary
    let remaining = if sorted_groups.len() > top_n {
        &sorted_groups[top_n..]
    } else {
        &[]
    };
    let remaining_count: u64 = remaining.iter().map(|(_, _, _, _, c)| *c).sum();
    let remaining_bytes: i64 = remaining.iter().map(|(_, _, _, b, _)| *b).sum();

    let sample_n = GC_ROOT_SAMPLE_N.min(remaining.len());
    let step = if remaining.len() > sample_n {
        remaining.len() / sample_n
    } else {
        1
    };
    let remaining_samples: Vec<ss_stats::ref_chain::GcRootMeta> = remaining
        .iter()
        .step_by(step)
        .take(GC_ROOT_SAMPLE_N)
        .filter_map(|&(_, _, root_idx, _, _)| root_metas.get(&root_idx).cloned())
        .collect();

    // Step 5: Build garbage info
    let mut garbage_class_map: HashMap<u32, (u64, i64)> = HashMap::new();
    for rank in 0..store.len() {
        let rec = match store.rec(rank) {
            Some(r) => r,
            None => continue,
        };
        if rec.kind == 3 {
            continue;
        }
        if !visited.get(rank) && root_of[rank as usize] == GC_ROOT_GARBAGE_SENTINEL {
            let entry = entry_of(heap, rec);
            let shallow = compute_obj_shallow(&entry, oop_size, header_size);
            let e = garbage_class_map.entry(entry.class_idx).or_insert((0, 0));
            e.0 += 1;
            e.1 += shallow;
        }
    }
    let mut garbage_top: Vec<(String, u64, i64)> = garbage_class_map
        .into_iter()
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

#[allow(clippy::too_many_arguments)]
fn build_single_root_tree(
    current_idx: u32,
    root_idx: u32,
    store: &ObjectStore<'_>,
    root_of: &[u32],
    field_pool: &mut FieldPool,
    buf: &mut Vec<u8>,
    oop_size: i32,
    header_size: i64,
    heap: &HeapDump,
    depth: usize,
    root_retained_bytes: i64,
) -> ss_stats::ref_chain::GcRootTreeNode {
    let entry = match store.entry(current_idx) {
        Some(e) => e,
        None => {
            return ss_stats::ref_chain::GcRootTreeNode {
                field_name: None,
                class_name: "unknown".to_string(),
                self_count: 0,
                self_bytes: 0,
                retained_bytes: 0,
                retained_count: 0,
                children: Vec::new(),
                collapsed_count: 0,
                collapsed_bytes: 0,
            }
        }
    };
    let self_bytes = compute_obj_shallow(&entry, oop_size, header_size);

    let mut node = ss_stats::ref_chain::GcRootTreeNode {
        field_name: None,
        class_name: get_class_display(entry.class_idx, heap),
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
    store.for_each_ref(current_idx, field_pool, buf, &mut |child, kind, class_idx, fk| {
        if kind == 3 {
            return;
        }
        if root_of[child as usize] != root_idx {
            return;
        }
        child_groups.entry((fk, class_idx)).or_default().push(child);
    });

    let shallow_of = |idx: u32| -> i64 {
        store
            .entry(idx)
            .map(|e| compute_obj_shallow(&e, oop_size, header_size))
            .unwrap_or(0)
    };

    // Sort groups by total shallow size descending
    let mut groups: Vec<((u32, u32), Vec<u32>)> = child_groups.into_iter().collect();
    groups.sort_by(|a, b| {
        let sa: i64 = a.1.iter().map(|&idx| shallow_of(idx)).sum();
        let sb: i64 = b.1.iter().map(|&idx| shallow_of(idx)).sum();
        sb.cmp(&sa)
    });

    // Prune threshold: don't prune at shallow depths (direct field refs are
    // always shown), then progressively prune tiny groups.
    let min_group_bytes: i64 = if depth < 2 {
        0
    } else {
        (root_retained_bytes / 10000).max(256) as i64
    };

    let max_children = GC_ROOT_MAX_CHILDREN;
    for (i, ((field_key, _class_idx), members)) in groups.iter().enumerate() {
        let group_bytes: i64 = members.iter().map(|&idx| shallow_of(idx)).sum();
        let group_count = members.len() as u64;

        if group_bytes < min_group_bytes {
            node.collapsed_count += group_count;
            node.collapsed_bytes += group_bytes;
            node.retained_bytes += group_bytes;
            node.retained_count += group_count;
            continue;
        }

        if i < max_children {
            let first_member = members[0];
            let mut child_node = build_single_root_tree(
                first_member,
                root_idx,
                store,
                root_of,
                field_pool,
                buf,
                oop_size,
                header_size,
                heap,
                depth + 1,
                root_retained_bytes,
            );
            child_node.field_name = Some(field_pool.get(*field_key).to_string());
            child_node.self_count = group_count;
            child_node.self_bytes = group_bytes;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_array_shallow_size_uses_element_width() {
        let int_array = ObjectEntry {
            obj_id: 1,
            class_idx: 0,
            body_size: 0,
            kind: 1,
            element_size: 4,
            array_length: 100,
        };
        // align8(header 12 + 4 bytes * 100) = 416
        assert_eq!(compute_obj_shallow(&int_array, 4, 12), 416);

        let byte_array = ObjectEntry {
            obj_id: 2,
            class_idx: 0,
            body_size: 0,
            kind: 1,
            element_size: 1,
            array_length: 100,
        };
        // align8(header 12 + 4-byte length + 1 byte * 100) = 120
        assert_eq!(compute_obj_shallow(&byte_array, 4, 12), 120);
    }
}
