use std::sync::Mutex;

/// Report for DirectByteBuffer off-heap memory analysis.
pub struct DirectBufferReport {
    pub total_count: usize,
    pub total_capacity: i64,
    pub per_class: Vec<DirectBufferClassEntry>,
    /// Sample object IDs (HPROF addresses) of DirectByteBuffer instances
    /// (top N by capacity) for per-instance reference chain rendering.
    pub sample_obj_ids: Vec<u64>,
}

#[derive(Clone)]
pub struct DirectBufferClassEntry {
    pub class_name: String,
    pub count: usize,
    pub total_capacity: i64,
}

/// Thread-safe wrapper for accumulating DirectByteBuffer results.
pub struct DirectBufferCollector {
    inner: Mutex<DirectBufferReport>,
}

impl DirectBufferCollector {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(DirectBufferReport {
                total_count: 0,
                total_capacity: 0,
                per_class: Vec::new(),
                sample_obj_ids: Vec::new(),
            }),
        }
    }

    pub fn set_results(
        &self,
        count: usize,
        capacity: i64,
        per_class: Vec<DirectBufferClassEntry>,
        sample_obj_ids: Vec<u64>,
    ) {
        let mut r = self.inner.lock().unwrap();
        r.total_count = count;
        r.total_capacity = capacity;
        r.per_class = per_class;
        r.sample_obj_ids = sample_obj_ids;
    }

    pub fn get_report(&self) -> DirectBufferReport {
        let r = self.inner.lock().unwrap();
        DirectBufferReport {
            total_count: r.total_count,
            total_capacity: r.total_capacity,
            per_class: r.per_class.clone(),
            sample_obj_ids: r.sample_obj_ids.clone(),
        }
    }
}
