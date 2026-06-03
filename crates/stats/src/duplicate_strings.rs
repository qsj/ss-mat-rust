use std::sync::Mutex;

/// Aggregate report for string deduplication analysis.
pub struct StringDupReport {
    pub total_strings: usize,
    pub unique_strings: usize,
    pub duplicate_group_count: usize,
    pub wasted_bytes: i64,
    pub top_groups: Vec<DupStringGroup>,
}

/// One group of duplicate String objects sharing the same content.
#[derive(Clone)]
pub struct DupStringGroup {
    pub content: String,
    pub count: usize,
    pub wasted_bytes: i64,
}

/// Thread-safe wrapper for accumulating string dup results.
pub struct StringDupCollector {
    inner: Mutex<StringDupReport>,
}

impl StringDupCollector {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(StringDupReport {
                total_strings: 0,
                unique_strings: 0,
                duplicate_group_count: 0,
                wasted_bytes: 0,
                top_groups: Vec::new(),
            }),
        }
    }

    pub fn set_results(&self, total: usize, unique: usize, groups: Vec<DupStringGroup>) {
        let mut r = self.inner.lock().unwrap();
        r.total_strings = total;
        r.unique_strings = unique;
        r.duplicate_group_count = groups.len();
        let mut wasted: i64 = 0;
        for g in &groups {
            wasted += g.wasted_bytes;
        }
        r.wasted_bytes = wasted;
        // Keep top 200 groups sorted by waste
        let mut sorted = groups;
        sorted.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));
        r.top_groups = sorted.into_iter().take(200).collect();
    }

    pub fn get_report(&self) -> StringDupReport {
        let r = self.inner.lock().unwrap();
        StringDupReport {
            total_strings: r.total_strings,
            unique_strings: r.unique_strings,
            duplicate_group_count: r.duplicate_group_count,
            wasted_bytes: r.wasted_bytes,
            top_groups: r.top_groups.clone(),
        }
    }
}
