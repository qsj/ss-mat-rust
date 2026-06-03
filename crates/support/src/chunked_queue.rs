/// Memory-efficient chunked queue for BFS traversal work items.
/// Replaces Java's ChunkedObjectQueue<T>.
pub struct ChunkedQueue<T> {
    chunks: Vec<Vec<T>>,
    chunk_capacity: usize,
}

impl<T> ChunkedQueue<T> {
    pub fn new(chunk_capacity: usize) -> Self {
        Self {
            chunks: Vec::new(),
            chunk_capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.chunks.is_empty() || self.chunks.last().unwrap().len() >= self.chunk_capacity {
            self.chunks.push(Vec::with_capacity(self.chunk_capacity));
        }
        self.chunks.last_mut().unwrap().push(item);
    }

    pub fn len(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty() || self.chunks.iter().all(|c| c.is_empty())
    }
}
