/// 14 problem types detected by the analysis engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProblemType {
    HighSelfMem,
    LongRefChain,
    DupStrings,
    DupPrimitiveArrays,
    DupObjectArrays,
    DupArrayLists,
    DupOtherObjects,
    BoxedNumbers,
    DirectByteBuffers,
    BadCollections,
    BadObjArrays,
    BadPrimArrays,
    BadWeakHm,
    HumongousObjs,
}

impl ProblemType {
    pub const COUNT: usize = 14;

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::HighSelfMem => "High self memory",
            Self::LongRefChain => "Long reference chains",
            Self::DupStrings => "Duplicate strings",
            Self::DupPrimitiveArrays => "Duplicate primitive arrays",
            Self::DupObjectArrays => "Duplicate object arrays",
            Self::DupArrayLists => "Duplicate array-based lists",
            Self::DupOtherObjects => "Duplicate objects",
            Self::BoxedNumbers => "Boxed numbers",
            Self::DirectByteBuffers => "Direct ByteBuffers",
            Self::BadCollections => "Bad collections",
            Self::BadObjArrays => "Bad object arrays",
            Self::BadPrimArrays => "Bad primitive arrays",
            Self::BadWeakHm => "WeakHashMap misuse",
            Self::HumongousObjs => "Humongous objects",
        }
    }
}

/// One bad object array record, captured during the linear scan so the
/// report can later filter to live-only using BFS `visited` (matching Java,
/// which only counts bad arrays reached during the live traversal with
/// `processingLiveObjects=true`).
#[derive(Clone, Debug)]
pub struct BadArrayRecord {
    pub obj_id: u64,
    pub class_idx: u32,
    /// 0=len-zero, 1=len-one, 2=all-null, 3=one-nonnull, 4=sparse.
    pub kind: u8,
    pub size: i64,
}

/// One bad primitive array record. Captured during the linear scan, then
/// filtered to live objects after BFS (matching Java's processPrimitiveArray,
/// which runs during the live traversal).
#[derive(Clone)]
pub struct BadPrimArrayRecord {
    pub obj_id: u64,
    pub class_idx: u32,
    /// 0=len-zero, 1=len-one, 2=all-zero, 3=trailing-zeroes, 4=sparse,
    /// 5=zero-high-bits.
    pub kind: u8,
    /// Full shallow size of the array body.
    pub size: i64,
    /// Java's overhead for this problem category.
    pub waste: i64,
}

pub const PRIM_ARRAY_LEN_ZERO: u8 = 0;
pub const PRIM_ARRAY_LEN_ONE: u8 = 1;
pub const PRIM_ARRAY_ALL_ZERO: u8 = 2;
pub const PRIM_ARRAY_TRAIL_ZEROES: u8 = 3;
pub const PRIM_ARRAY_SPARSE: u8 = 4;
pub const PRIM_ARRAY_ZERO_HIGH_BITS: u8 = 5;

pub fn prim_array_problem_name(kind: u8) -> &'static str {
    match kind {
        PRIM_ARRAY_LEN_ZERO => "0-length",
        PRIM_ARRAY_LEN_ONE => "1-length",
        PRIM_ARRAY_ALL_ZERO => "empty (all zeroes)",
        PRIM_ARRAY_TRAIL_ZEROES => "trail-0s",
        PRIM_ARRAY_SPARSE => "sparse",
        PRIM_ARRAY_ZERO_HIGH_BITS => "hi-bits-0s",
        _ => "unknown",
    }
}

/// Java `CollectionProblemKind` ordinals. The scan stores these directly so
/// the debug dump can be compared field-by-field with Java's
/// `PerClassProblemStats.problemKindCounts/problemKindSizes`.
pub const COLLECTION_EMPTY: u8 = 0;
pub const COLLECTION_LEN_ZERO: u8 = 1;
pub const COLLECTION_LEN_ONE: u8 = 2;
pub const COLLECTION_SINGLE_ELEM: u8 = 3;
pub const COLLECTION_SMALL: u8 = 4;
pub const COLLECTION_TRAIL_ZEROES: u8 = 5;
pub const COLLECTION_SPARSE: u8 = 6;
pub const COLLECTION_ZERO_HIGH_BITS: u8 = 7;
pub const COLLECTION_MANY_NULLS: u8 = 8;
pub const COLLECTION_KIND_COUNT: usize = 9;
/// Sentinel for "no problem recorded" (Java's `kind == null`).
pub const COLLECTION_KIND_NONE: u8 = u8::MAX;

/// Java `CollectionProblemKind.toString()`.
pub fn collection_problem_name(kind: u8) -> &'static str {
    match kind {
        COLLECTION_EMPTY => "empty",
        COLLECTION_LEN_ZERO => "0-length",
        COLLECTION_LEN_ONE => "1-length",
        COLLECTION_SINGLE_ELEM => "1-elem",
        COLLECTION_SMALL => "small",
        COLLECTION_TRAIL_ZEROES => "trail-0s",
        COLLECTION_SPARSE => "sparse",
        COLLECTION_ZERO_HIGH_BITS => "hi-bits-0s",
        COLLECTION_MANY_NULLS => "many nulls",
        _ => "-",
    }
}

/// Java `CollectionProblemKind.name()` (enum constant name).
pub fn collection_problem_enum_name(kind: u8) -> &'static str {
    match kind {
        COLLECTION_EMPTY => "EMPTY",
        COLLECTION_LEN_ZERO => "LEN_ZERO",
        COLLECTION_LEN_ONE => "LEN_ONE",
        COLLECTION_SINGLE_ELEM => "SINGLE_ELEM",
        COLLECTION_SMALL => "SMALL",
        COLLECTION_TRAIL_ZEROES => "TRAIL_ZEROES",
        COLLECTION_SPARSE => "SPARSE",
        COLLECTION_ZERO_HIGH_BITS => "ZERO_HIGH_BITS",
        COLLECTION_MANY_NULLS => "MANY_NULLS",
        _ => "-",
    }
}

/// One bad collection record, captured during the scan. `capacity` is
/// resolved post-pass from the backing-array lookup; SPARSE detection uses
/// `size <= capacity/3`.
#[derive(Clone)]
pub struct BadCollectionRecord {
    pub obj_id: u64,
    pub class_idx: u32,
    pub size: i32,
    /// Aligned shallow body of the container object itself.
    pub container_bytes: i64,
    /// Backing-array object id (e.g. HashMap.table / ArrayList.elementData),
    /// resolved to a length post-pass for SPARSE/SMALL waste computation.
    /// 0 when no backing field was found.
    pub backing_array_id: u64,
    /// WeakHashMap's `queue` object (ReferenceQueue). Java's
    /// WeakHashMapIterator adds the queue and its lock object bodies.
    pub queue_id: u64,
    /// LinkedList's `first` node, used for Java's MANY_NULLS null-ratio check.
    pub chain_head: u64,
    /// ConcurrentSkipListMap's `head` Index, used to derive `getSize()`.
    pub cslm_head: u64,
    /// ConcurrentHashMap's `counterCells` array, used to derive `getSize()`.
    pub counter_cells: u64,
    /// Waste computed so far. EMPTY/SINGLE_ELEM set this to the shallow body
    /// during the scan; SPARSE adds (capacity-size)*refSize post-pass.
    pub waste: i64,
    /// Java `CollectionProblemKind` ordinal (EMPTY/SINGLE_ELEM/SMALL/SPARSE/
    /// MANY_NULLS), or `COLLECTION_KIND_NONE` when the post-pass found no
    /// problem. Filled by linear_scan; the report ignores it.
    pub kind: u8,
    /// Java `ContainerInstance.getBodySize()` at classification time; forensic
    /// only (used by the bad_collections debug dump).
    pub body: i64,
}

/// Fixed-width result encoding. Store iteration is fallible and preserves order.
pub trait StoredRecord: Sized {
    const WIDTH: usize;
    fn encode(&self, raw: &mut [u8]);
    fn decode(raw: &[u8]) -> Self;
}

/// Bytes one positional read moves. Fixed, so the cache a store keeps does not
/// depend on the record count.
const STORE_READ_BLOCK_BYTES: usize = 64 * 1024;

/// Records per read block for width-byte records; never zero.
fn block_records(width: usize) -> usize {
    (STORE_READ_BLOCK_BYTES / width).max(1)
}

/// Mutable per-store I/O state. get/set take &self, hence the RefCell.
struct StoreIo {
    /// window_len * T::WIDTH valid bytes starting at record window_start.
    read: Vec<u8>,
    window_start: u64,
    window_len: usize,
    /// One record wide, so encoding a push or a set allocates nothing.
    scratch: Vec<u8>,
}

pub struct RecordStore<T: StoredRecord> {
    file: std::fs::File,
    count: u64,
    io: std::cell::RefCell<StoreIo>,
    marker: std::marker::PhantomData<T>,
}

impl<T: StoredRecord> RecordStore<T> {
    pub fn new() -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var_os("SS_MAT_TEMP_DIR").filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
        loop {
            let path = base.join(format!("ss-mat-results-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            match std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path) {
                Ok(file) => {
                    std::fs::remove_file(path)?;
                    return Ok(Self {
                        file,
                        count: 0,
                        io: std::cell::RefCell::new(StoreIo {
                            read: vec![0; block_records(T::WIDTH) * T::WIDTH],
                            window_start: 0,
                            window_len: 0,
                            scratch: vec![0; T::WIDTH],
                        }),
                        marker: std::marker::PhantomData,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }

    /// Byte offset of index, or the overflow error every accessor reports.
    fn offset(index: u64) -> std::io::Result<u64> {
        index.checked_mul(T::WIDTH as u64).ok_or_else(|| std::io::Error::other("record offset overflow"))
    }

    /// Read the block that starts at index into the window cache.
    fn fill_window(&self, io: &mut StoreIo, index: u64) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let n = block_records(T::WIDTH).min((self.count - index) as usize);
        self.file.read_exact_at(&mut io.read[..n * T::WIDTH], Self::offset(index)?)?;
        io.window_start = index;
        io.window_len = n;
        Ok(())
    }

    pub fn push(&mut self, record: T) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let io = self.io.get_mut();
        record.encode(&mut io.scratch);
        self.file.write_all_at(&io.scratch, Self::offset(self.count)?)?;
        self.count += 1;
        Ok(())
    }

    /// Rewrite one record in place. The change is visible to every later get and
    /// iter, including when the record sits in the cached read window.
    pub fn set(&self, index: u64, record: T) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        if index >= self.count { return Err(std::io::Error::other("result index")); }
        let mut cache = self.io.borrow_mut();
        let io: &mut StoreIo = &mut cache;
        record.encode(&mut io.scratch);
        self.file.write_all_at(&io.scratch, Self::offset(index)?)?;
        if io.window_len > 0 && index >= io.window_start && index - io.window_start < io.window_len as u64 {
            let at = (index - io.window_start) as usize * T::WIDTH;
            io.read[at..at + T::WIDTH].copy_from_slice(&io.scratch);
        }
        Ok(())
    }

    pub fn get(&self, index: u64) -> std::io::Result<T> {
        if index >= self.count { return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "result index")); }
        let mut cache = self.io.borrow_mut();
        let io: &mut StoreIo = &mut cache;
        if io.window_len > 0 && index >= io.window_start && index - io.window_start < io.window_len as u64 {
            let at = (index - io.window_start) as usize * T::WIDTH;
            return Ok(T::decode(&io.read[at..at + T::WIDTH]));
        }
        self.fill_window(io, index)?;
        Ok(T::decode(&io.read[..T::WIDTH]))
    }

    pub fn len(&self) -> u64 { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }
    pub fn iter(&self) -> RecordIter<'_, T> { RecordIter { store: self, at: 0 } }
}

pub struct RecordIter<'a, T: StoredRecord> { store: &'a RecordStore<T>, at: u64 }
impl<T: StoredRecord> Iterator for RecordIter<'_, T> {
    type Item = std::io::Result<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.at == self.store.count { return None; }
        let result = self.store.get(self.at);
        self.at += 1;
        Some(result)
    }
}

impl StoredRecord for BadArrayRecord {
    const WIDTH: usize = 24;
    fn encode(&self, r: &mut [u8]) {
        r[..8].copy_from_slice(&self.obj_id.to_le_bytes());
        r[8..12].copy_from_slice(&self.class_idx.to_le_bytes()); r[12] = self.kind;
        r[16..24].copy_from_slice(&self.size.to_le_bytes());
    }
    fn decode(r: &[u8]) -> Self { Self { obj_id: u64::from_le_bytes(r[..8].try_into().unwrap()),
        class_idx: u32::from_le_bytes(r[8..12].try_into().unwrap()), kind: r[12],
        size: i64::from_le_bytes(r[16..24].try_into().unwrap()) } }
}
impl StoredRecord for BadPrimArrayRecord {
    const WIDTH: usize = 32;
    fn encode(&self, r: &mut [u8]) {
        BadArrayRecord { obj_id: self.obj_id, class_idx: self.class_idx, kind: self.kind, size: self.size }.encode(r);
        r[24..32].copy_from_slice(&self.waste.to_le_bytes());
    }
    fn decode(r: &[u8]) -> Self {
        let a = BadArrayRecord::decode(r);
        Self { obj_id: a.obj_id, class_idx: a.class_idx, kind: a.kind, size: a.size,
            waste: i64::from_le_bytes(r[24..32].try_into().unwrap()) }
    }
}

#[cfg(test)]
mod record_store_tests {
    use super::*;
    #[test]
    fn repeated_reads_and_truncation_propagate() {
        let mut store = RecordStore::new().unwrap();
        store.push(BadArrayRecord { obj_id: 7, class_idx: 3, kind: 2, size: 19 }).unwrap();
        for _ in 0..2 { assert_eq!(store.iter().next().unwrap().unwrap().obj_id, 7); }
        // The record is in the read window, so truncating the file behind it is
        // not observable; a store that still reads the file surfaces the error.
        let mut fresh = RecordStore::new().unwrap();
        fresh.push(BadArrayRecord { obj_id: 7, class_idx: 3, kind: 2, size: 19 }).unwrap();
        fresh.file.set_len(2).unwrap();
        assert_eq!(fresh.iter().next().unwrap().unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    }
}

pub fn sort_record_keys(input: &std::path::Path, output: &std::path::Path, width: usize, keys: &[usize]) -> std::io::Result<()> {
    use std::io::{Read, Write};
    let key = keys[0];
    let sorted = output.with_extension(format!("level{}", keys.len()));
    ss_support::external_sort::ExternalSorter::new(width, 8 << 20)
        .sort_file(input, &sorted, |r| u64::from_le_bytes(r[key..key+8].try_into().unwrap()))?;
    if keys.len() == 1 { std::fs::rename(sorted, output)?; return Ok(()); }
    let mut reader = std::io::BufReader::new(std::fs::File::open(sorted)?);
    let mut writer = std::io::BufWriter::new(std::fs::File::create(output)?);
    let count = reader.get_ref().metadata()?.len() / width as u64;
    let mut read = 0;
    let mut next = None;
    let run = output.with_extension(format!("run{}", keys.len()));
    let run_sorted = output.with_extension(format!("sorted{}", keys.len()));
    while read < count || next.is_some() {
        let first = if let Some(raw) = next.take() { raw } else {
            let mut raw = vec![0; width]; reader.read_exact(&mut raw)?; read += 1; raw
        };
        if read == count { writer.write_all(&first)?; break; }
        let mut raw = vec![0; width]; reader.read_exact(&mut raw)?; read += 1;
        if first[key..key+8] != raw[key..key+8] { writer.write_all(&first)?; next = Some(raw); continue; }
        let mut run_writer = std::io::BufWriter::new(std::fs::File::create(&run)?);
        run_writer.write_all(&first)?;
        loop {
            run_writer.write_all(&raw)?;
            if read == count { break; }
            reader.read_exact(&mut raw)?; read += 1;
            if first[key..key+8] != raw[key..key+8] { next = Some(raw); break; }
        }
        run_writer.flush()?;
        sort_record_keys(&run, &run_sorted, width, &keys[1..])?;
        std::io::copy(&mut std::fs::File::open(&run_sorted)?, &mut writer)?;
    }
    writer.flush()
}

impl StoredRecord for BadCollectionRecord {
    const WIDTH:usize=96;
    fn encode(&self,r:&mut[u8]) {
        let values=[self.obj_id,self.class_idx as u64,self.size as u64,self.container_bytes as u64,self.backing_array_id,
            self.queue_id,self.chain_head,self.cslm_head,self.counter_cells,self.waste as u64,self.kind as u64,self.body as u64];
        for (raw,n) in r.chunks_exact_mut(8).zip(values){raw.copy_from_slice(&n.to_le_bytes());}
    }
    fn decode(r:&[u8])->Self {
        let n=|i|u64::from_le_bytes(r[i..i+8].try_into().unwrap());
        Self{obj_id:n(0),class_idx:n(8) as u32,size:n(16) as i32,container_bytes:n(24) as i64,
            backing_array_id:n(32),queue_id:n(40),chain_head:n(48),cslm_head:n(56),counter_cells:n(64),
            waste:n(72) as i64,kind:n(80) as u8,body:n(88) as i64}
    }
}
impl StoredRecord for (i64,i32) {
    const WIDTH:usize=16;
    fn encode(&self,r:&mut[u8]){r[..8].copy_from_slice(&self.0.to_le_bytes());r[8..12].copy_from_slice(&self.1.to_le_bytes());}
    fn decode(r:&[u8])->Self{(i64::from_le_bytes(r[..8].try_into().unwrap()),i32::from_le_bytes(r[8..12].try_into().unwrap()))}
}

#[cfg(test)]
mod ordering_tests {
    use super::*;
    #[test]
    fn composite_sort_retains_duplicates_and_complete_secondary_order() {
        use std::io::{Read,Write};
        let base=std::env::temp_dir().join(format!("ss-mat-sort-key-test-{}",std::process::id()));
        std::fs::create_dir(&base).unwrap();
        let input=base.join("input");let output=base.join("output");
        let mut file=std::fs::File::create(&input).unwrap();
        let mut expected=Vec::new();
        for i in (0..2000u64).rev(){let row=(i%3,i%17,i);expected.push(row);for n in [row.0,row.1,row.2]{file.write_all(&n.to_le_bytes()).unwrap();}}
        sort_record_keys(&input,&output,24,&[0,8,16]).unwrap();expected.sort();
        let mut file=std::fs::File::open(output).unwrap();
        for row in expected{let mut raw=[0;24];file.read_exact(&mut raw).unwrap();let n=|i|u64::from_le_bytes(raw[i..i+8].try_into().unwrap());assert_eq!((n(0),n(8),n(16)),row);}
        std::fs::remove_dir_all(base).unwrap();
    }
}
