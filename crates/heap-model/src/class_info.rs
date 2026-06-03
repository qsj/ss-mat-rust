use std::sync::Arc;

/// Metadata for a Java class in the heap dump.
/// Replaces Java's JavaClassInfo.
#[derive(Debug, Clone)]
pub struct JavaClassInfo {
    pub name: Arc<str>,
    pub class_object_id: u64,
    pub super_class_id: u64,
    pub instance_size: i32,
    pub instance_field_count: i32,
    pub instance_fields: Vec<FieldDesc>,
    pub static_fields: Vec<FieldDesc>,
    pub category: ClassCategory,
    /// Index into the class table
    pub index: u32,
}

/// A field descriptor with name and type.
#[derive(Debug, Clone)]
pub struct FieldDesc {
    pub name: Arc<str>,
    pub type_code: u8,
}

/// Category classification for classes based on their role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassCategory {
    String,
    StringValue, // char[] / byte[] (Java 9+)
    Class,
    Thread,
    ClassLoader,
    Collection,
    Map,
    Array,
    BoxedNumber,
    DirectByteBuffer,
    WeakReference,
    SunMiscCleaner,
    JdkInternalRefCleaner,
    ThreadGroup,
    Other,
}

