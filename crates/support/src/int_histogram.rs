/// Histogram builder for size distribution analysis.
/// Replaces Java's IntHistogramBuilder.
pub struct IntHistogram {
    buckets: Vec<(i64, i64)>, // (threshold, count)
}

impl IntHistogram {
    pub fn new(thresholds: &[i64]) -> Self {
        let mut buckets: Vec<(i64, i64)> = thresholds.iter().map(|&t| (t, 0)).collect();
        buckets.push((i64::MAX, 0)); // overflow bucket
        Self { buckets }
    }

    pub fn add(&mut self, value: i64) {
        for bucket in &mut self.buckets {
            if value <= bucket.0 {
                bucket.1 += 1;
                break;
            }
        }
    }

    pub fn buckets(&self) -> &[(i64, i64)] {
        &self.buckets
    }
}
