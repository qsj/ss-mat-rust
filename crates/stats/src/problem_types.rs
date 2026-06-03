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
