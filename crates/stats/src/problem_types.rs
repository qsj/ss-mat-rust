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
