// Progress reporting utilities.
// Will be expanded as the pipeline is implemented.

pub struct Progress {
    total: u64,
    current: std::sync::atomic::AtomicU64,
    message: String,
}

impl Progress {
    pub fn new(message: &str, total: u64) -> Self {
        Self {
            total,
            current: std::sync::atomic::AtomicU64::new(0),
            message: message.to_string(),
        }
    }

    pub fn increment(&self) {
        let current = self.current.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if current % 100_000 == 0 {
            eprintln!("{}: {}/{}", self.message, current, self.total);
        }
    }

    pub fn done(&self) {
        let current = self.current.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("{}: {}/{} (done)", self.message, current, self.total);
    }
}
