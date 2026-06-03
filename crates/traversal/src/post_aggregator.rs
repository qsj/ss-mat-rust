use std::sync::Arc;

use ss_heap_model::heap_dump::HeapDump;
use ss_stats::engine::StatsEngine;

/// Post-traversal aggregator: computes per-class retained sizes
/// and dominated sizes after BFS completes.
pub struct PostTraversalAggregator {
    heap: Arc<HeapDump>,
    stats: Arc<StatsEngine>,
}

impl PostTraversalAggregator {
    pub fn new(heap: Arc<HeapDump>, stats: Arc<StatsEngine>) -> Self {
        Self { heap, stats }
    }

    /// Finalize all post-traversal calculations.
    pub fn finalize(&self) {
        // Future: compute dominated sizes, finalize per-class problem stats
    }
}
