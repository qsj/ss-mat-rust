use std::sync::Mutex;

/// Aggregate report for string deduplication analysis.
pub struct StringDupReport {
    pub total_strings: usize,
    pub unique_strings: usize,
    pub duplicate_group_count: usize,
    pub wasted_bytes: i64,
    /// Java `StringContentDeduplicator.stringAlignedBodySize` (aligned body of
    /// the java.lang.String class object). Used only by the forensic dump.
    pub string_body_size: i64,
    pub top_groups: Vec<DupStringGroup>,
}

/// One group of duplicate String objects sharing the same content.
#[derive(Clone)]
pub struct DupStringGroup {
    pub content: String,
    pub count: usize,
    pub wasted_bytes: i64,
    /// Number of distinct backing char[]/byte[] arrays referenced by this
    /// group (Java `StringDeduplicationStats.backingArrayCount`).
    pub backing_array_count: usize,
    /// Sample object IDs (HPROF addresses) of String instances in this group.
    /// Used by the report to render per-instance reference chains. Empty if
    /// the scanner didn't capture obj IDs.
    pub sample_obj_ids: Vec<u64>,
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
                string_body_size: 0,
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
        // Keep top 200 groups sorted by waste for the report.
        let mut sorted = groups;
        sorted.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));
        // Forensic dump: keep every duplicate group, not just the report's
        // top 200. Inert unless SS_MAT_DEBUG_DIR is set.
        if crate::debug_dump::enabled() {
            crate::debug_dump::record_dup_string_groups(sorted.clone());
        }
        r.top_groups = sorted.into_iter().take(200).collect();
    }

    pub fn set_streamed_results(&self, total: usize, unique: usize, count: usize, wasted: i64, top: Vec<DupStringGroup>) {
        let mut r = self.inner.lock().unwrap();
        r.total_strings=total; r.unique_strings=unique; r.duplicate_group_count=count;
        r.wasted_bytes=wasted; r.top_groups=top;
    }

    pub fn get_report(&self) -> StringDupReport {
        let r = self.inner.lock().unwrap();
        StringDupReport {
            total_strings: r.total_strings,
            unique_strings: r.unique_strings,
            duplicate_group_count: r.duplicate_group_count,
            wasted_bytes: r.wasted_bytes,
            string_body_size: r.string_body_size,
            top_groups: r.top_groups.clone(),
        }
    }

    /// Record the aligned String body size used by the waste formula.
    pub fn set_string_body_size(&self, size: i64) {
        self.inner.lock().unwrap().string_body_size = size;
    }
}

#[cfg(test)]
mod streamed_tests {
    use super::*;
    #[test]
    fn streamed_totals_are_not_limited_to_rendered_prefix() {
        for count in [0usize, 201, 10000] {
            let collector=StringDupCollector::new();
            let top=(0..count.min(200)).map(|_| DupStringGroup { content:String::new(),count:2,
                wasted_bytes:7,backing_array_count:1,sample_obj_ids:Vec::new() }).collect();
            collector.set_streamed_results(count*2,count,count,(count*7) as i64,top);
            let report=collector.get_report();
            assert_eq!(report.duplicate_group_count,count);
            assert_eq!(report.wasted_bytes,(count*7) as i64);
            assert_eq!(report.top_groups.len(),count.min(200));
        }
    }
}
