/// Probabilistic sampler for reducing the number of GC roots in reports.
/// Replaces Java's ProbabilisticSampler.
pub struct ProbabilisticSampler {
    sample_rate: f64,
}

impl ProbabilisticSampler {
    pub fn new(sample_rate: f64) -> Self {
        Self { sample_rate }
    }

    pub fn should_sample(&self) -> bool {
        if self.sample_rate >= 1.0 {
            return true;
        }
        rand_sample(self.sample_rate)
    }
}

fn rand_sample(rate: f64) -> bool {
    // Simple deterministic sampling based on rate
    // In production, use a proper PRNG
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos as f64 / u32::MAX as f64) < rate
}
