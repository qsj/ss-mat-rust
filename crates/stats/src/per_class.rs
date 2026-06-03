use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64};

/// Per-class statistics accumulated during linear scan and BFS traversal.
/// One instance per class in the heap dump.
pub struct PerClassStats {
    pub class_index: u32,
    pub class_name: String,
    pub instance_size: i32,

    // Instance counts and sizes (populated during linear scan)
    pub instance_count: AtomicU64,
    pub total_instance_bytes: AtomicI64,

    // Array statistics (populated during linear scan)
    pub array_live_count: AtomicU32,
    pub array_live_bytes: AtomicI64,
    pub array_garbage_count: AtomicU32,
    pub array_garbage_bytes: AtomicI64,

    // Instance garbage tracking (populated during BFS)
    pub instance_garbage_count: AtomicU32,
    pub instance_garbage_bytes: AtomicI64,

    // Retained size (populated during BFS traversal)
    pub retained_size: AtomicI64,

    // Problem overhead accumulators
    pub boxed_number_overhead: AtomicI64,
    pub duplicate_string_bytes: AtomicI64,
    pub collection_overhead: AtomicI64,
    pub humongous_count: AtomicU32,
    pub humongous_bytes: AtomicI64,
    pub humongous_overhead: AtomicI64,
}

impl PerClassStats {
    pub fn new(class_index: u32, class_name: String, instance_size: i32) -> Self {
        Self {
            class_index,
            class_name,
            instance_size,
            instance_count: AtomicU64::new(0),
            total_instance_bytes: AtomicI64::new(0),
            array_live_count: AtomicU32::new(0),
            array_live_bytes: AtomicI64::new(0),
            array_garbage_count: AtomicU32::new(0),
            array_garbage_bytes: AtomicI64::new(0),
            instance_garbage_count: AtomicU32::new(0),
            instance_garbage_bytes: AtomicI64::new(0),
            retained_size: AtomicI64::new(0),
            boxed_number_overhead: AtomicI64::new(0),
            duplicate_string_bytes: AtomicI64::new(0),
            collection_overhead: AtomicI64::new(0),
            humongous_count: AtomicU32::new(0),
            humongous_bytes: AtomicI64::new(0),
            humongous_overhead: AtomicI64::new(0),
        }
    }

    pub fn add_instance(&self, shallow_size: i64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.instance_count.fetch_add(1, Relaxed);
        self.total_instance_bytes.fetch_add(shallow_size, Relaxed);
    }

    pub fn add_array(&self, length: i32, element_size: i32, is_garbage: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        let size = array_shallow_size(length, element_size);
        if is_garbage {
            self.array_garbage_count.fetch_add(1, Relaxed);
            self.array_garbage_bytes.fetch_add(size, Relaxed);
        } else {
            self.array_live_count.fetch_add(1, Relaxed);
            self.array_live_bytes.fetch_add(size, Relaxed);
        }
    }
}

/// Calculate shallow size of an array: header + length * element_size, aligned.
fn array_shallow_size(length: i32, element_size: i32) -> i64 {
    // JVM array header: 12 bytes (mark word + klass pointer) + 4 bytes (length)
    let header = 16i64;
    let data_size = length as i64 * element_size as i64;
    let raw = header + data_size;
    // Align to 8 bytes
    (raw + 7) & !7
}
