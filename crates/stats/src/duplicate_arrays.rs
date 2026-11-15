use std::sync::Mutex;

/// Report for duplicate arrays/lists of a particular category.
pub struct DupArrayReport {
    pub total_count: usize,
    pub unique_count: usize,
    pub duplicate_group_count: usize,
    pub wasted_bytes: i64,
    pub per_class: Vec<DupArrayClassEntry>,
    pub top_groups: Vec<DupArrayGroup>,
    /// Java's `buildDupReport` returns null when the summed waste is zero; the
    /// report then renders the section header only.
    pub report_present: bool,
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
    /// Java group id (S15); primitive-array groups leave it 0.
    pub group_id: u32,
    /// Representative object address (S15); first sample otherwise.
    pub representative_obj_id: u64,
    /// Brief description for display (e.g., length or first few elements)
    pub description: String,
    /// Sample object IDs (HPROF addresses) of array instances in this group.
    /// Used by the report to render per-instance reference chains.
    pub sample_obj_ids: Vec<u64>,
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
                report_present: false,
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
        r.report_present = r.wasted_bytes > 0;
        r.per_class = per_class;
        let mut sorted = top_groups;
        sorted.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));
        r.top_groups = sorted.into_iter().take(100).collect();
    }

    /// Java S15 semantics: totals come from the detector because
    /// `ClassDominatedSizeComputer` redefines them (`totalGroupCount` counts
    /// duplicate groups plus non-enumerated instances; `duplicateGroupCount`
    /// is the number of enumerated groups).
    pub fn set_obj_results(
        &self,
        total_count: usize,
        unique_count: usize,
        duplicate_group_count: usize,
        wasted_bytes: i64,
        per_class: Vec<DupArrayClassEntry>,
        top_groups: Vec<DupArrayGroup>,
    ) {
        let mut r = self.inner.lock().unwrap();
        r.total_count = total_count;
        r.unique_count = unique_count;
        r.duplicate_group_count = duplicate_group_count;
        r.wasted_bytes = wasted_bytes;
        r.report_present = wasted_bytes > 0;
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
            report_present: r.report_present,
        }
    }
}
