use std::sync::Mutex;

/// Report for duplicate arrays/lists of a particular category.
pub struct DupArrayReport {
    pub total_count: usize,
    pub unique_count: usize,
    pub duplicate_group_count: usize,
    pub wasted_bytes: i64,
    pub per_class: Vec<DupArrayClassEntry>,
    pub top_groups: Vec<DupArrayGroup>,
}

#[derive(Clone)]
pub struct DupArrayClassEntry {
    pub class_name: String,
    pub instance_count: usize,
    pub unique_count: usize,
    pub wasted_bytes: i64,
}

#[derive(Clone)]
pub struct DupArrayGroup {
    pub class_name: String,
    pub count: usize,
    pub wasted_bytes: i64,
    pub size_bytes: i64,
    /// Brief description for display (e.g., length or first few elements)
    pub description: String,
}

/// Thread-safe wrapper for accumulating duplicate array results.
pub struct DupArrayCollector {
    inner: Mutex<DupArrayReport>,
}

impl DupArrayCollector {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(DupArrayReport {
                total_count: 0,
                unique_count: 0,
                duplicate_group_count: 0,
                wasted_bytes: 0,
                per_class: Vec::new(),
                top_groups: Vec::new(),
            }),
        }
    }

    pub fn set_results(
        &self,
        total: usize,
        unique: usize,
        per_class: Vec<DupArrayClassEntry>,
        top_groups: Vec<DupArrayGroup>,
    ) {
        let mut r = self.inner.lock().unwrap();
        r.total_count = total;
        r.unique_count = unique;
        r.duplicate_group_count = per_class.len();
        r.wasted_bytes = per_class.iter().map(|e| e.wasted_bytes).sum();
        r.per_class = per_class;
        let mut sorted = top_groups;
        sorted.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));
        r.top_groups = sorted.into_iter().take(100).collect();
    }

    pub fn get_report(&self) -> DupArrayReport {
        let r = self.inner.lock().unwrap();
        DupArrayReport {
            total_count: r.total_count,
            unique_count: r.unique_count,
            duplicate_group_count: r.duplicate_group_count,
            wasted_bytes: r.wasted_bytes,
            per_class: r.per_class.clone(),
            top_groups: r.top_groups.clone(),
        }
    }
}
