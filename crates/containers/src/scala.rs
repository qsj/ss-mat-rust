use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// Scala ArrayBuffer — backed by Object[].
pub struct ScalaArrayBufferDescriptor;

impl ContainerDescriptor for ScalaArrayBufferDescriptor {
    fn container_class_name(&self) -> &str { "scala/collection/mutable/ArrayBuffer" }
    fn is_map(&self) -> bool { false }

    fn create_instance(
        &self, _class_idx: u32, _fields: &[u8], _field_descriptors: &[FieldDescRef], _id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>> {
        Some(Box::new(SimpleInstance { size: 0, capacity: 0 }))
    }
}

/// Scala HashMap — hash table based map.
pub struct ScalaHashMapDescriptor;

impl ContainerDescriptor for ScalaHashMapDescriptor {
    fn container_class_name(&self) -> &str { "scala/collection/mutable/HashMap" }
    fn is_map(&self) -> bool { true }

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
