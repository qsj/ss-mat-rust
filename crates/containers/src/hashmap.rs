use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// Descriptor for hash-table-based maps: HashMap, Hashtable, LinkedHashMap, WeakHashMap.
pub struct HashTableBasedMapDescriptor {
    class_name: &'static str,
    size_field_name: &'static str,
    table_field_name: &'static str,
}

impl HashTableBasedMapDescriptor {
    pub fn hashmap() -> Self {
        Self { class_name: "java/util/HashMap", size_field_name: "size", table_field_name: "table" }
    }
    pub fn linked_hashmap() -> Self {
        Self { class_name: "java/util/LinkedHashMap", size_field_name: "size", table_field_name: "table" }
    }
    pub fn hashtable() -> Self {
        Self { class_name: "java/util/Hashtable", size_field_name: "count", table_field_name: "table" }
    }
    pub fn weak_hashmap() -> Self {
        Self { class_name: "java/util/WeakHashMap", size_field_name: "size", table_field_name: "table" }
    }
}

impl ContainerDescriptor for HashTableBasedMapDescriptor {
    fn container_class_name(&self) -> &str { self.class_name }
    fn is_map(&self) -> bool { true }

    fn create_instance(
        &self, _class_idx: u32, fields: &[u8], field_descriptors: &[FieldDescRef], id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>> {
        let id_bytes = id_size as usize;
        let mut size_val: i32 = 0;
        let mut offset = 0usize;

        for fd in field_descriptors {
            match fd.type_code {
                10 => {
                    if fd.name == self.size_field_name {
                        size_val = read_i32(fields, offset).unwrap_or(0);
                    }
                    offset += 4;
                }
                2 => { offset += id_bytes; }
                _ => { offset += field_size(fd.type_code, id_bytes); }
            }
        }

        Some(Box::new(HashTableMapInstance { size: size_val, table_capacity: 0 }))
    }
}

struct HashTableMapInstance { size: i32, table_capacity: i32 }

impl ContainerInstance for HashTableMapInstance {
    fn is_map(&self) -> bool { true }
    fn size(&self) -> i32 { self.size }
    fn backing_capacity(&self) -> i32 { self.table_capacity }
}

/// HashSet / LinkedHashSet — delegates to backing HashMap.
pub struct HashSetDescriptor { class_name: &'static str }

impl HashSetDescriptor {
    pub fn hashset() -> Self { Self { class_name: "java/util/HashSet" } }
    pub fn linked_hashset() -> Self { Self { class_name: "java/util/LinkedHashSet" } }
}

impl ContainerDescriptor for HashSetDescriptor {
    fn container_class_name(&self) -> &str { self.class_name }
    fn is_map(&self) -> bool { false }

    fn create_instance(
        &self, _class_idx: u32, _fields: &[u8], _field_descriptors: &[FieldDescRef], _id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>> {
        // Size is determined by the backing HashMap — needs cross-reference during traversal
        Some(Box::new(SimpleCollectionInstance { size: 0, capacity: 0 }))
    }
}

struct SimpleCollectionInstance { size: i32, capacity: i32 }

impl ContainerInstance for SimpleCollectionInstance {
    fn is_map(&self) -> bool { false }
    fn size(&self) -> i32 { self.size }
    fn backing_capacity(&self) -> i32 { self.capacity }
}

fn read_i32(data: &[u8], offset: usize) -> Option<i32> {
    if offset + 4 > data.len() { return None; }
    Some(i32::from_be_bytes(data[offset..offset + 4].try_into().ok()?))
}

fn field_size(type_code: u8, id_bytes: usize) -> usize {
    match type_code {
        2 => id_bytes, 4 => 1, 5 => 2, 6 => 4, 7 => 8, 8 => 1, 9 => 2, 10 => 4, 11 => 8, _ => 0,
    }
}
