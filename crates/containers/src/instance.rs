/// Runtime representation of a collection being walked.
/// Created by a ContainerDescriptor from instance field data.
pub trait ContainerInstance {
    /// Whether this is a map (key-value) or a plain collection.
    fn is_map(&self) -> bool;

    /// Logical size of the collection (number of entries/elements).
    fn size(&self) -> i32;

    /// Overhead: the difference between backing storage capacity and logical size.
    /// For a HashMap: table.length - size.
    /// For an ArrayList: elementData.length - size.
    fn overhead(&self) -> i32 {
        self.backing_capacity() - self.size()
    }

    /// Capacity of the backing storage (table array length, element array length, etc.)
    fn backing_capacity(&self) -> i32;

    /// Overhead bytes: (backing_capacity - size) * reference_size + container_object_header.
    /// Simplified: just reports overhead * 8 as an approximation.
    fn overhead_bytes(&self) -> i64 {
        let wasted_slots = self.overhead();
        if wasted_slots > 0 {
            wasted_slots as i64 * 8 // reference size
        } else {
            0
        }
    }
}
