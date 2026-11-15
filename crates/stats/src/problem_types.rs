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
#[derive(Clone)]
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
