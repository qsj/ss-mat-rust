use std::sync::atomic::{AtomicI64, AtomicI32, Ordering};

/// Trait for reference chain tree nodes used in retained size computation.
/// Replaces the Java ReferenceNode class hierarchy.
pub trait ReferenceNode: Send + Sync {
    fn class_name(&self) -> &str;
    fn retained_size(&self) -> i64;
    fn object_count(&self) -> i32;
    fn add_to_size_and_count(&self, size: i64, count: i32);
}

/// Atomic accumulator for retained size and object count.
pub struct SizeAndCount {
    size: AtomicI64,
    count: AtomicI32,
}

impl SizeAndCount {
    pub fn new() -> Self {
        Self {
            size: AtomicI64::new(0),
            count: AtomicI32::new(0),
        }
    }

    pub fn add(&self, size: i64, count: i32) {
        self.size.fetch_add(size, Ordering::Relaxed);
        self.count.fetch_add(count, Ordering::Relaxed);
    }

    pub fn get_size(&self) -> i64 {
        self.size.load(Ordering::Relaxed)
    }

    pub fn get_count(&self) -> i32 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Default for SizeAndCount {
    fn default() -> Self {
        Self::new()
    }
}
