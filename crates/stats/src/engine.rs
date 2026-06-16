use std::sync::atomic::{AtomicI64, AtomicU64};
use std::sync::Mutex;

use ss_heap_model::class_info::JavaClassInfo;
use ss_heap_model::heap_dump::{FrameLocals, ThreadInfo};

use crate::direct_buffer::DirectBufferCollector;
use crate::duplicate_arrays::DupArrayCollector;
use crate::duplicate_strings::StringDupCollector;
use crate::jvm_info::JvmInfoCollector;
use crate::per_class::PerClassStats;
use crate::problem_types::ProblemType;
use crate::ref_chain::{GcRootStore, RefChainStore};
use crate::traversal_data::TraversalData;

/// Central statistics accumulator. Populated by linear scan and BFS traversal.
pub struct StatsEngine {
    pub per_class: Vec<PerClassStats>,
    pub total_instances: AtomicU64,
    pub total_arrays: AtomicU64,
    pub total_shallow_size: AtomicI64,
    pub total_retained_size: AtomicI64,
    pub total_garbage_count: AtomicU64,
    pub total_garbage_bytes: AtomicI64,
    pub problem_totals: [AtomicI64; ProblemType::COUNT],
    pub string_dup: StringDupCollector,
    pub dup_prim_arrays: DupArrayCollector,
    pub dup_obj_arrays: DupArrayCollector,
    pub direct_buffers: DirectBufferCollector,
    pub jvm_info: JvmInfoCollector,
    pub thread_info: Mutex<Vec<ThreadInfo>>,
    pub oom_frame_locals: Mutex<Vec<FrameLocals>>,
    pub ref_chains: Mutex<RefChainStore>,
    pub gc_roots: Mutex<GcRootStore>,
    pub traversal_data: Mutex<Option<TraversalData>>,

    // Bad object array counters (populated during linear scan)
    pub empty_obj_array_count: AtomicU64,
    pub empty_obj_array_bytes: AtomicI64,
    pub single_elem_obj_array_count: AtomicU64,
    pub single_elem_obj_array_bytes: AtomicI64,
    pub all_null_obj_array_count: AtomicU64,
    pub all_null_obj_array_bytes: AtomicI64,
    pub one_nonnull_obj_array_count: AtomicU64,
    pub one_nonnull_obj_array_bytes: AtomicI64,
    pub sparse_obj_array_count: AtomicU64,
    pub sparse_obj_array_bytes: AtomicI64,
    // Bad primitive array counters
    pub empty_prim_array_count: AtomicU64,
    pub empty_prim_array_bytes: AtomicI64,
    pub trailing_zero_prim_array_count: AtomicU64,
    pub trailing_zero_prim_array_bytes: AtomicI64,
    // Bad array sample obj_ids (HPROF addresses) for per-instance ref chain
    // rendering. Each Vec is capped at 50 entries.
    pub empty_prim_array_samples: Mutex<Vec<u64>>,
    pub trailing_zero_prim_array_samples: Mutex<Vec<u64>>,
    pub empty_obj_array_samples: Mutex<Vec<u64>>,
    pub single_elem_obj_array_samples: Mutex<Vec<u64>>,
    pub all_null_obj_array_samples: Mutex<Vec<u64>>,
    pub one_nonnull_obj_array_samples: Mutex<Vec<u64>>,
    pub sparse_obj_array_samples: Mutex<Vec<u64>>,
}

impl StatsEngine {
    pub fn new(classes: &[JavaClassInfo]) -> Self {
        let per_class: Vec<PerClassStats> = classes
            .iter()
            .map(|c| {
                PerClassStats::new(
                    c.index,
                    c.name.to_string(),
                    c.instance_size,
                )
            })
            .collect();

        Self {
            per_class,
            total_instances: AtomicU64::new(0),
            total_arrays: AtomicU64::new(0),
            total_shallow_size: AtomicI64::new(0),
            total_retained_size: AtomicI64::new(0),
            total_garbage_count: AtomicU64::new(0),
            total_garbage_bytes: AtomicI64::new(0),
            problem_totals: Default::default(),
            string_dup: StringDupCollector::new(),
            dup_prim_arrays: DupArrayCollector::new(),
            dup_obj_arrays: DupArrayCollector::new(),
            direct_buffers: DirectBufferCollector::new(),
            jvm_info: JvmInfoCollector::new(),
            thread_info: Mutex::new(Vec::new()),
            oom_frame_locals: Mutex::new(Vec::new()),
            ref_chains: Mutex::new(RefChainStore::default()),
            gc_roots: Mutex::new(GcRootStore::default()),
            traversal_data: Mutex::new(None),
            empty_obj_array_count: AtomicU64::new(0),
            empty_obj_array_bytes: AtomicI64::new(0),
            single_elem_obj_array_count: AtomicU64::new(0),
            single_elem_obj_array_bytes: AtomicI64::new(0),
            all_null_obj_array_count: AtomicU64::new(0),
            all_null_obj_array_bytes: AtomicI64::new(0),
            one_nonnull_obj_array_count: AtomicU64::new(0),
            one_nonnull_obj_array_bytes: AtomicI64::new(0),
            sparse_obj_array_count: AtomicU64::new(0),
            sparse_obj_array_bytes: AtomicI64::new(0),
            empty_prim_array_count: AtomicU64::new(0),
            empty_prim_array_bytes: AtomicI64::new(0),
            trailing_zero_prim_array_count: AtomicU64::new(0),
            trailing_zero_prim_array_bytes: AtomicI64::new(0),
            empty_prim_array_samples: Mutex::new(Vec::new()),
            trailing_zero_prim_array_samples: Mutex::new(Vec::new()),
            empty_obj_array_samples: Mutex::new(Vec::new()),
            single_elem_obj_array_samples: Mutex::new(Vec::new()),
            all_null_obj_array_samples: Mutex::new(Vec::new()),
            one_nonnull_obj_array_samples: Mutex::new(Vec::new()),
            sparse_obj_array_samples: Mutex::new(Vec::new()),
        }
    }

    pub fn get_per_class(&self, class_index: u32) -> Option<&PerClassStats> {
        self.per_class.get(class_index as usize)
    }

    /// Print a summary of top classes by instance count.
    pub fn print_summary(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        eprintln!("\n=== Stats Summary ===");
        eprintln!(
            "Total instances: {}",
            self.total_instances.load(Relaxed)
        );
        eprintln!("Total arrays: {}", self.total_arrays.load(Relaxed));
        eprintln!(
            "Total shallow size: {} bytes",
            self.total_shallow_size.load(Relaxed)
        );

        // Sort by total bytes (instance + array) descending
        let mut indexed: Vec<(i64, &PerClassStats)> = self
            .per_class
            .iter()
            .map(|s| {
                let total = s.total_instance_bytes.load(Relaxed)
                    + s.array_live_bytes.load(Relaxed)
                    + s.retained_size.load(Relaxed);
                (total, s)
            })
            .filter(|(c, _)| *c > 0)
            .collect();
        indexed.sort_by(|a, b| b.0.cmp(&a.0));

        eprintln!("\nTop 20 classes by total size:");
        for (total_bytes, stats) in indexed.iter().take(20) {
            let inst = stats.instance_count.load(Relaxed);
            let arr = stats.array_live_count.load(Relaxed);
            eprintln!(
                "  {:>14} bytes  ({} inst, {} arr)  {}",
                total_bytes, inst, arr, stats.class_name
            );
        }
    }
}
