use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// EnumMap — array-backed map with enum keys.
pub struct EnumMapDescriptor;

impl ContainerDescriptor for EnumMapDescriptor {
    fn container_class_name(&self) -> &str { "java/util/EnumMap" }
    fn is_map(&self) -> bool { true }

    fn create_instance(
        &self, _class_idx: u32, _fields: &[u8], _field_descriptors: &[FieldDescRef], _id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>> {
        Some(Box::new(SimpleInstance { size: 0, capacity: 0 }))
    }
}

/// EnumSet — bit-vector based set for enums.
pub struct EnumSetDescriptor;

impl ContainerDescriptor for EnumSetDescriptor {
    fn container_class_name(&self) -> &str { "java/util/RegularEnumSet" }
    fn is_map(&self) -> bool { false }

    fn create_instance(
        &self, _class_idx: u32, _fields: &[u8], _field_descriptors: &[FieldDescRef], _id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>> {
        Some(Box::new(SimpleInstance { size: 0, capacity: 0 }))
    }
}

struct SimpleInstance { size: i32, capacity: i32 }

impl ContainerInstance for SimpleInstance {
    fn is_map(&self) -> bool { false }
    fn size(&self) -> i32 { self.size }
    fn backing_capacity(&self) -> i32 { self.capacity }
}
