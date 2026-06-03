use crate::instance::ContainerInstance;

/// A container descriptor describes a known Java collection type.
pub trait ContainerDescriptor: Send + Sync {
    /// The Java class name this descriptor handles.
    fn container_class_name(&self) -> &str;

    /// Whether this is a map type (key-value pairs).
    fn is_map(&self) -> bool;

    /// Create a ContainerInstance from the given instance's field values.
    fn create_instance(
        &self,
        class_idx: u32,
        fields: &[u8],
        field_descriptors: &[FieldDescRef],
        id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>>;
}

/// Reference to a field descriptor: name and type code.
pub struct FieldDescRef<'a> {
    pub name: &'a str,
    pub type_code: u8,
}
