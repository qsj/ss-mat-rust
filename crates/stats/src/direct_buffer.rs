use std::sync::Mutex;

/// Report for DirectByteBuffer off-heap memory analysis.
pub struct DirectBufferReport {
    pub total_count: usize,
    pub total_capacity: i64,
    pub per_class: Vec<DirectBufferClassEntry>,
    /// Sample object IDs (HPROF addresses) of DirectByteBuffer instances
    /// (top N by capacity) for per-instance reference chain rendering.
    pub sample_obj_ids: Vec<u64>,
    /// Per-buffer records (address-deduped capacity + object id), so the
    /// report crate can split reachable/unreachable using BFS `visited`.
    /// Only buffers with non-zero deduped capacity are kept.
    pub buffers: Vec<DirectBufferRecord>,
}

/// One DirectByteBuffer with its address-deduped capacity and object id.
#[derive(Clone)]
pub struct DirectBufferRecord {
    pub obj_id: u64,
    /// Capacity after address dedup (0 for views sharing another buffer's
    /// native memory).
    pub capacity: i64,
    pub class_name: String,
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
                buffers: Vec::new(),
            }),
        }
    }

    pub fn set_results(
        &self,
        count: usize,
        capacity: i64,
        per_class: Vec<DirectBufferClassEntry>,
        sample_obj_ids: Vec<u64>,
        buffers: Vec<DirectBufferRecord>,
    ) {
        let mut r = self.inner.lock().unwrap();
        r.total_count = count;
        r.total_capacity = capacity;
        r.per_class = per_class;
        r.sample_obj_ids = sample_obj_ids;
        r.buffers = buffers;
    }

    pub fn get_report(&self) -> DirectBufferReport {
        let r = self.inner.lock().unwrap();
        DirectBufferReport {
            total_count: r.total_count,
            total_capacity: r.total_capacity,
            per_class: r.per_class.clone(),
            sample_obj_ids: r.sample_obj_ids.clone(),
            buffers: r.buffers.clone(),
        }
    }
}
