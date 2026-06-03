use std::sync::Arc;

/// Describes an instance or static field of a class.
#[derive(Debug, Clone)]
pub struct FieldDescriptor {
    pub name: Arc<str>,
    pub type_code: u8,
}

impl FieldDescriptor {
    /// Returns the size in bytes for this field's type.
    pub fn type_size(&self) -> usize {
        crate::primitive_type::type_size(self.type_code)
    }

    /// Returns true if this field holds an object reference.
    pub fn is_object(&self) -> bool {
        self.type_code == 2
    }
}
