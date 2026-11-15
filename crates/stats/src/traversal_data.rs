//! Post-traversal data handed to the HTML report.
//!
//! Everything here used to be one in-RAM array per object: an object table
//! (24 B/object), a parent array plus a parent-edge array (8 B/object) and a
//! `Vec<bool>` visited array (1 B/object) — roughly 100 MB on a 2.79 M-object
//! dump, on top of the 78 MB edge table the BFS also kept. The report only ever
//! looks at a handful of objects, so the data now lives behind cheap handles:
//!
//! * object metadata is answered by the dump's on-disk object directory
//!   (address → offset/class/kind/length), which the parser already builds and
//!   which every other phase shares;
//! * the BFS parent log is a dense 8-byte-per-object lookup file;
//! * liveness is a bit set (1 bit per object);
//! * the dump's own object order (the order Java's object index uses, and the
//!   order samples must be drawn in) is a 4-byte-per-object table on disk.

use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

use ss_heap_model::heap_dump::HeapDump;
use ss_support::obj_dir::ObjRec;

/// Per-object metadata, decoded on demand from the object directory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjectEntry {
    pub obj_id: u64,
    pub class_idx: u32,
    pub body_size: i32,
    /// Object type: 0=instance, 1=primitive array, 2=object array, 3=class object
    pub kind: u8,
    /// Bytes per array element: primitive width for kind=1, oop size for kind=2,
    /// 0 for instances and class objects.
    pub element_size: u8,
    pub array_length: i32,
}

/// Map an object-directory kind to the kind this crate's callers use.
///
/// The directory numbers arrays the other way round from the rest of the
/// pipeline: it reports 1 for object arrays and 2 for primitive arrays, while
/// `ObjectEntry` has always used 1 for primitive and 2 for object arrays.
pub fn object_kind(dir_kind: u8) -> u8 {
    match dir_kind {
        1 => 2,
        2 => 1,
        other => other,
    }
}

/// Decode the directory record of one object into report metadata.
///
/// `body_size` for instances is the class's HPROF `instance_size`; the object
/// directory does not store it because it is a per-class constant.
pub fn entry_of(heap: &HeapDump, rec: ObjRec) -> ObjectEntry {
    let body_size = if rec.kind == 0 {
        heap.classes
            .get(rec.class_idx as usize)
            .map(|c| c.instance_size)
            .unwrap_or(0)
    } else {
        0
    };
    let kind = object_kind(rec.kind);
    let element_size = if kind == 2 {
        // Object arrays are sized by the JVM's oop width, not by the dump's
        // identifier width (a compressed-oop dump has 8-byte ids and 4-byte oops).
        heap.oop_size
    } else {
        rec.elem
    };
    ObjectEntry {
        obj_id: rec.addr,
        class_idx: rec.class_idx,
        body_size,
        kind,
        element_size,
        array_length: rec.len as i32,
    }
}

/// Field name pool: compact storage of field names used in edges.
/// Index 0 = "[elements]" (array elements), 1 = "[static]" (static field).
pub struct FieldPool {
    pub names: Vec<String>,
    /// Name to key, so interning stays constant-time: the traversal interns once
    /// per reference and a linear scan over the pool dominated the BFS phase.
    index: std::collections::HashMap<String, u32>,
}

impl FieldPool {
    pub fn new() -> Self {
        let names = vec!["[elements]".to_string(), "[static]".to_string()];
        let index = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u32))
            .collect();
        Self { names, index }
    }
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(&idx) = self.index.get(name) {
            return idx;
        }
        let idx = self.names.len() as u32;
        self.names.push(name.to_string());
        self.index.insert(name.to_string(), idx);
        idx
    }
    pub fn get(&self, key: u32) -> &str {
        &self.names[key as usize]
    }
}

impl Default for FieldPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Edge in the object graph carrying field info. Only used by transient,
/// per-object decodes now that the BFS reads references straight from the dump.
#[derive(Clone)]
pub struct OutEdge {
    pub target: u32,
    pub field_key: u32, // index into FieldPool
}

/// Parent value meaning "the BFS never reached this object".
pub const NO_PARENT: u32 = u32::MAX;

/// One record of the BFS parent log: (parent rank, field key).
pub const PARENT_REC_SIZE: usize = 8;

/// Liveness bit set, one bit per object rank.
pub struct LiveSet {
    words: Vec<u64>,
    len: u32,
}

impl LiveSet {
    pub fn new(len: u32) -> Self {
        Self {
            words: vec![0u64; (len as usize).div_ceil(64)],
            len,
        }
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn get(&self, idx: u32) -> bool {
        let i = idx as usize;
        i < self.len as usize && (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, idx: u32) {
        let i = idx as usize;
        if i < self.len as usize {
            self.words[i / 64] |= 1u64 << (i % 64);
        }
    }

    /// Number of set bits.
    pub fn count(&self) -> u64 {
        self.words.iter().map(|w| w.count_ones() as u64).sum()
    }
}

fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(not(unix))]
    {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = file.try_clone()?;
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }
}

/// Dump order table: `addr_rank[hprof_rank]` for every object.
///
/// The dump lists class dumps first and then objects roughly by address, while
/// the object directory is strictly address-sorted; the two orders differ, and
/// sample selection must follow the dump's order to stay identical to the
/// previous in-memory object table.
pub struct DumpOrder {
    file: File,
    len: u32,
}

/// Ranks read per bulk step while streaming the dump order.
const ORDER_CHUNK_RECS: usize = 1 << 16;

impl DumpOrder {
    pub fn open(path: &Path, len: u32) -> io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            len,
        })
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Address rank of the `dump_rank`-th object in dump order.
    pub fn addr_rank(&self, dump_rank: u32) -> Option<u32> {
        if dump_rank >= self.len {
            return None;
        }
        let mut buf = [0u8; 4];
        read_exact_at(&self.file, &mut buf, dump_rank as u64 * 4).ok()?;
        Some(u32::from_le_bytes(buf))
    }

    /// Stream `(dump_rank, addr_rank)` pairs in dump order.
    pub fn for_each<F: FnMut(u32, u32)>(&self, mut f: F) -> io::Result<()> {
        let mut buf = vec![0u8; ORDER_CHUNK_RECS * 4];
        let mut base = 0u32;
        while base < self.len {
            let recs = ORDER_CHUNK_RECS.min((self.len - base) as usize);
            let bytes = &mut buf[..recs * 4];
            read_exact_at(&self.file, bytes, base as u64 * 4)?;
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                let mut b = [0u8; 4];
                b.copy_from_slice(chunk);
                f(base + i as u32, u32::from_le_bytes(b));
            }
            base += recs as u32;
        }
        Ok(())
    }
}

/// Dense BFS parent log: one 8-byte record per object rank.
pub struct ParentStore {
    file: File,
    len: u32,
}

impl ParentStore {
    pub fn open(path: &Path, len: u32) -> io::Result<Self> {
        Ok(Self {
            file: File::open(path)?,
            len,
        })
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `(parent rank, field key)` of one object; `NO_PARENT` when unreached.
    pub fn parent_of(&self, idx: u32) -> (u32, u32) {
        if idx >= self.len {
            return (NO_PARENT, 0);
        }
        let mut buf = [0u8; PARENT_REC_SIZE];
        if read_exact_at(&self.file, &mut buf, idx as u64 * PARENT_REC_SIZE as u64).is_err() {
            return (NO_PARENT, 0);
        }
        let mut p = [0u8; 4];
        p.copy_from_slice(&buf[0..4]);
        let mut f = [0u8; 4];
        f.copy_from_slice(&buf[4..8]);
        (u32::from_le_bytes(p), u32::from_le_bytes(f))
    }
}

/// Persistent snapshot of BFS data exposed for post-traversal rendering
/// (e.g., reference chains in the HTML report).
///
/// Populated by `HeapObjectTraverser::run_traversal` and stored in
/// `StatsEngine::traversal_data` so the report crate can walk object→root
/// chains on demand without re-running BFS.
pub struct TraversalData {
    /// The dump whose object directory answers metadata queries.
    pub heap: Arc<HeapDump>,
    /// Number of objects, i.e. records in the object directory.
    pub object_count: u32,
    /// Dump order → address rank.
    pub order: DumpOrder,
    /// BFS parent log.
    pub parent: ParentStore,
    /// GC root that first reached each object, indexed by address rank.
    pub root_of: Vec<u32>,
    /// Objects the BFS reached, indexed by address rank.
    pub visited: LiveSet,
    pub field_pool: FieldPool,
}

impl TraversalData {
    /// Metadata of one object, by address rank.
    pub fn object_at(&self, rank: u32) -> Option<ObjectEntry> {
        let dir = self.heap.obj_dir.as_ref()?;
        let rec = dir.get(rank as u64).ok()?;
        Some(entry_of(&self.heap, rec))
    }

    /// Object address (HPROF id) of one address rank.
    pub fn obj_id_of(&self, rank: u32) -> Option<u64> {
        self.object_at(rank).map(|e| e.obj_id)
    }

    /// Address rank of an object address.
    pub fn rank_of_id(&self, obj_id: u64) -> Option<u32> {
        let dir = self.heap.obj_dir.as_ref()?;
        dir.lookup(obj_id).ok()?.map(|(rank, _)| rank as u32)
    }

    pub fn is_live(&self, rank: u32) -> bool {
        self.visited.get(rank)
    }

    /// True when the object address was reached by the BFS.
    pub fn is_live_id(&self, obj_id: u64) -> bool {
        match self.rank_of_id(obj_id) {
            Some(rank) => self.visited.get(rank),
            None => false,
        }
    }

    /// Stream every object in the dump's own order.
    pub fn for_each_in_dump_order<F>(&self, mut f: F)
    where
        F: FnMut(u32, u32, ObjectEntry),
    {
        let dir = match self.heap.obj_dir.as_ref() {
            Some(d) => d,
            None => return,
        };
        let _ = self.order.for_each(|dump_rank, addr_rank| {
            if let Ok(rec) = dir.get(addr_rank as u64) {
                f(dump_rank, addr_rank, entry_of(&self.heap, rec));
            }
        });
    }

    /// First `limit` object addresses of one class, in the dump's own order.
    ///
    /// This is the order the old in-memory object table enumerated in, and the
    /// report's sample sets depend on it.
    pub fn sample_ids_by_class(&self, class_idx: u32, limit: usize) -> Vec<u64> {
        let mut out = Vec::with_capacity(limit.min(64));
        if limit == 0 {
            return out;
        }
        let dir = match self.heap.obj_dir.as_ref() {
            Some(d) => d,
            None => return out,
        };
        let _ = self.order.for_each(|_, addr_rank| {
            if out.len() >= limit {
                return;
            }
            if let Ok(rec) = dir.get(addr_rank as u64) {
                if rec.class_idx == class_idx {
                    out.push(rec.addr);
                }
            }
        });
        out
    }
}
