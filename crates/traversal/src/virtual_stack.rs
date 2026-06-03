/// VirtualStackTraverser: stack-based deep reference chain traversal.
/// Used for computing reference chain depth from GC roots.
pub struct VirtualStackTraverser;

impl VirtualStackTraverser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for VirtualStackTraverser {
    fn default() -> Self {
        Self::new()
    }
}
