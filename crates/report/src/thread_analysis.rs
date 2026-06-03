/// Thread analysis helper for identifying OOM-causing threads.
pub struct ThreadAnalysisHelper;

impl ThreadAnalysisHelper {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ThreadAnalysisHelper {
    fn default() -> Self {
        Self::new()
    }
}
