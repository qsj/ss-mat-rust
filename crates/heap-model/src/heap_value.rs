/// JVM heap value — replaces 12 separate Java value classes with a single enum.
#[derive(Debug, Clone, PartialEq)]
pub enum HeapValue {
    ObjectRef(Option<u32>), // None = null
    Boolean(bool),          // type code 4
    Char(u16),              // type code 5
    Float(f32),             // type code 6
    Double(f64),            // type code 7
    Byte(i8),               // type code 8
    Short(i16),             // type code 9
    Int(i32),               // type code 10
    Long(i64),              // type code 11
}

impl HeapValue {
    /// Returns the size in bytes for this value type when stored in a heap dump.
    pub fn type_size(&self) -> usize {
        match self {
            Self::ObjectRef(_) => 4, // index-based, but original is pointer-sized
            Self::Boolean(_) => 1,
            Self::Char(_) => 2,
            Self::Float(_) => 4,
            Self::Double(_) => 8,
            Self::Byte(_) => 1,
            Self::Short(_) => 2,
            Self::Int(_) => 4,
            Self::Long(_) => 8,
        }
    }

    /// Extract the object reference index, if this is an ObjectRef.
    pub fn as_object_ref(&self) -> Option<Option<u32>> {
        match self {
            Self::ObjectRef(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Extract as i64 for numeric comparisons.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Byte(v) => Some(*v as i64),
            Self::Short(v) => Some(*v as i64),
            Self::Int(v) => Some(*v as i64),
            Self::Long(v) => Some(*v),
            Self::Char(v) => Some(*v as i64),
            Self::Boolean(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }
}
