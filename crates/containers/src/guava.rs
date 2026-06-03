use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// Guava ImmutableMap — backed by an Object[] of alternating key-value pairs.
pub struct GuavaImmutableMapDescriptor;

impl ContainerDescriptor for GuavaImmutableMapDescriptor {
    fn container_class_name(&self) -> &str { "com/google/common/collect/RegularImmutableMap" }
    fn is_map(&self) -> bool { true }

    fn create_instance(
        &self, _class_idx: u32, _fields: &[u8], _field_descriptors: &[FieldDescRef], _id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>> {
        Some(Box::new(SimpleInstance { size: 0, capacity: 0 }))
    }
}

struct SimpleInstance { size: i32, capacity: i32 }

impl ContainerInstance for SimpleInstance {
    fn is_map(&self) -> bool { true }
    fn size(&self) -> i32 { self.size }
    fn backing_capacity(&self) -> i32 { self.capacity }
}
