use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// ConcurrentHashMap v7 (Segment-based) and v8 (CAS-based single table).
pub struct ChmDescriptor { class_name: &'static str }

impl ChmDescriptor {
    pub fn chm_v7() -> Self { Self { class_name: "java/util/concurrent/ConcurrentHashMap" } }
    pub fn chm_v8() -> Self { Self { class_name: "java/util/concurrent/ConcurrentHashMap" } }
}

impl ContainerDescriptor for ChmDescriptor {
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
                    if fd.name == "size" || fd.name == "count" {
                        size_val = read_i32(fields, offset).unwrap_or(0);
                    }
                    offset += 4;
                }
                2 => { offset += id_bytes; }
                _ => { offset += field_size(fd.type_code, id_bytes); }
            }
        }

        Some(Box::new(SimpleInstance { size: size_val, capacity: 0, is_map: true }))
    }
}

struct SimpleInstance { size: i32, capacity: i32, is_map: bool }

impl ContainerInstance for SimpleInstance {
    fn is_map(&self) -> bool { self.is_map }
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
