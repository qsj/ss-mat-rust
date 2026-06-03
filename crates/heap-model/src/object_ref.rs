/// An object reference in the heap dump, represented as a compressed index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObjectRef(pub u32);

impl ObjectRef {
    pub const NULL: ObjectRef = ObjectRef(u32::MAX);

    #[inline]
    pub fn is_null(&self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub fn index(&self) -> u32 {
        self.0
    }
}
