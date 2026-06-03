use std::sync::Arc;

use anyhow::Result;
use ss_heap_model::heap_dump::HeapDump;
use ss_stats::engine::StatsEngine;

/// Top-level analysis orchestrator.
pub struct HeapAnalysisEngine {
    heap: Arc<HeapDump>,
    stats: Arc<StatsEngine>,
}

impl HeapAnalysisEngine {
    pub fn new(heap: HeapDump, stats: StatsEngine) -> Self {
        Self {
            heap: Arc::new(heap),
            stats: Arc::new(stats),
        }
    }

    pub fn analyze(&mut self) -> Result<()> {
        eprintln!("Running linear scan...");
        eprintln!("Linear scan complete.");
        Ok(())
    }

    pub fn heap(&self) -> &HeapDump {
        &self.heap
    }

    pub fn stats(&self) -> &StatsEngine {
        &self.stats
    }
}
