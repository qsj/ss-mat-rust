use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// CompactLinkedHashMap / CompactLinkedHashSet — used by some frameworks.
pub struct CompactLinkedDescriptor { class_name: &'static str, is_map: bool }

impl CompactLinkedDescriptor {
    pub fn compact_linked_hash_map() -> Self {
        Self { class_name: "com/google/common/collect/CompactLinkedHashMap", is_map: true }
    }
    pub fn compact_linked_hash_set() -> Self {
        Self { class_name: "com/google/common/collect/CompactLinkedHashSet", is_map: false }
    }
}

impl ContainerDescriptor for CompactLinkedDescriptor {
    fn container_class_name(&self) -> &str { self.class_name }
    fn is_map(&self) -> bool { self.is_map }

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
