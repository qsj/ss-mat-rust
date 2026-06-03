use crate::descriptor::{ContainerDescriptor, FieldDescRef};
use crate::instance::ContainerInstance;

/// Descriptor for array-backed collections: ArrayList, Vector, Stack, CopyOnWriteArrayList.
/// These all have a "size" field and an "elementData" (or similar) backing array.
pub struct ArrayBackedDescriptor {
    class_name: &'static str,
    size_field_names: &'static [&'static str],
    array_field_names: &'static [&'static str],
}

impl ArrayBackedDescriptor {
    pub fn arraylist() -> Self {
        Self {
            class_name: "java/util/ArrayList",
            size_field_names: &["size"],
            array_field_names: &["elementData"],
        }
    }
    pub fn vector() -> Self {
        Self {
            class_name: "java/util/Vector",
            size_field_names: &["elementCount"],
            array_field_names: &["elementData"],
        }
    }
    pub fn stack() -> Self {
        Self {
            class_name: "java/util/Stack",
            size_field_names: &["elementCount"],
            array_field_names: &["elementData"],
        }
    }
    pub fn copy_on_write_arraylist() -> Self {
        Self {
            class_name: "java/util/concurrent/CopyOnWriteArrayList",
            size_field_names: &["size"],
            array_field_names: &["elements", "array"],
        }
    }
}

impl ContainerDescriptor for ArrayBackedDescriptor {
    fn container_class_name(&self) -> &str { self.class_name }
    fn is_map(&self) -> bool { false }

    fn create_instance(
        &self, _class_idx: u32, fields: &[u8], field_descriptors: &[FieldDescRef], id_size: u8,
    ) -> Option<Box<dyn ContainerInstance>> {
        let id_bytes = id_size as usize;
        let mut size_val: i32 = 0;
        let mut offset = 0usize;

        for fd in field_descriptors {
            match fd.type_code {
                10 => {
                    if self.size_field_names.contains(&fd.name) {
                        size_val = read_i32(fields, offset).unwrap_or(0);
                    }
                    offset += 4;
                }
                2 => { offset += id_bytes; }
                _ => { offset += field_size(fd.type_code, id_bytes); }
            }
        }

        Some(Box::new(ArrayBackedInstance { size: size_val, array_capacity: 0 }))
    }
}

struct ArrayBackedInstance { size: i32, array_capacity: i32 }

impl ContainerInstance for ArrayBackedInstance {
    fn is_map(&self) -> bool { false }
    fn size(&self) -> i32 { self.size }
    fn backing_capacity(&self) -> i32 { self.array_capacity }
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
