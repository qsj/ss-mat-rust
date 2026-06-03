use std::collections::HashMap;

use crate::descriptor::ContainerDescriptor;
use crate::array_backed::ArrayBackedDescriptor;
use crate::hashmap::{HashTableBasedMapDescriptor, HashSetDescriptor};
use crate::chm::ChmDescriptor;
use crate::treemap::TreeMapDescriptor;
use crate::linked_list::LinkedListDescriptor;
use crate::identity_hashmap::IdentityHashMapDescriptor;
use crate::skip_list::SkipListDescriptor;
use crate::enums::{EnumMapDescriptor, EnumSetDescriptor};
use crate::compact_linked::CompactLinkedDescriptor;

pub struct ContainerRegistry {
    descriptors: HashMap<&'static str, Box<dyn ContainerDescriptor>>,
}

impl ContainerRegistry {
    pub fn new() -> Self {
        let mut d: HashMap<&'static str, Box<dyn ContainerDescriptor>> = HashMap::new();

        d.insert("java/util/HashMap", Box::new(HashTableBasedMapDescriptor::hashmap()));
        d.insert("java/util/LinkedHashMap", Box::new(HashTableBasedMapDescriptor::linked_hashmap()));
        d.insert("java/util/Hashtable", Box::new(HashTableBasedMapDescriptor::hashtable()));
        d.insert("java/util/WeakHashMap", Box::new(HashTableBasedMapDescriptor::weak_hashmap()));
        d.insert("java/util/HashSet", Box::new(HashSetDescriptor::hashset()));
        d.insert("java/util/LinkedHashSet", Box::new(HashSetDescriptor::linked_hashset()));
        d.insert("java/util/ArrayList", Box::new(ArrayBackedDescriptor::arraylist()));
        d.insert("java/util/Vector", Box::new(ArrayBackedDescriptor::vector()));
        d.insert("java/util/Stack", Box::new(ArrayBackedDescriptor::stack()));
        d.insert("java/util/concurrent/CopyOnWriteArrayList", Box::new(ArrayBackedDescriptor::copy_on_write_arraylist()));
        d.insert("java/util/concurrent/ConcurrentHashMap", Box::new(ChmDescriptor::chm_v7()));
        d.insert("java/util/TreeMap", Box::new(TreeMapDescriptor::treemap()));
        d.insert("java/util/LinkedList", Box::new(LinkedListDescriptor::linkedlist()));
        d.insert("java/util/IdentityHashMap", Box::new(IdentityHashMapDescriptor));
        d.insert("java/util/concurrent/ConcurrentSkipListMap", Box::new(SkipListDescriptor::concurrent_skip_list_map()));
        d.insert("java/util/EnumMap", Box::new(EnumMapDescriptor));
        d.insert("java/util/RegularEnumSet", Box::new(EnumSetDescriptor));
        d.insert("com/google/common/collect/CompactLinkedHashMap", Box::new(CompactLinkedDescriptor::compact_linked_hash_map()));
        d.insert("com/google/common/collect/CompactLinkedHashSet", Box::new(CompactLinkedDescriptor::compact_linked_hash_set()));

        Self { descriptors: d }
    }

    pub fn lookup(&self, class_name: &str) -> Option<&dyn ContainerDescriptor> {
        self.descriptors.get(class_name).map(|b| b.as_ref())
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }
}

impl Default for ContainerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
