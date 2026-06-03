use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// TreeMap — red-black tree map. Has size field, root node with left/right entries.
pub struct TreeMapDescriptor { class_name: &'static str }

impl TreeMapDescriptor {
    pub fn treemap() -> Self { Self { class_name: "java/util/TreeMap" } }
}

impl ContainerDescriptor for TreeMapDescriptor {
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
                    if fd.name == "size" { size_val = read_i32(fields, offset).unwrap_or(0); }
                    offset += 4;
                }
                2 => { offset += id_bytes; }
                _ => { offset += field_size(fd.type_code, id_bytes); }
            }
        }

        Some(Box::new(SimpleInstance { size: size_val, capacity: size_val }))
    }
}

struct SimpleInstance { size: i32, capacity: i32 }

impl ContainerInstance for SimpleInstance {
    fn is_map(&self) -> bool { true }
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
