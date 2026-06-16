//! Advisor trait + supporting types.
//!
//! Each `Advisor` examines a `ProblemReportSet` (derived from `StatsEngine`)
//! and optionally returns an `Advice`. The registry runs all advisors and
//! the resulting advices are rendered to the "MOST IMPORTANT ISSUES" panel.

use ss_heap_model::heap_dump::HeapDump;
use ss_stats::engine::StatsEngine;
use ss_stats::direct_buffer::DirectBufferReport;
use ss_stats::duplicate_arrays::DupArrayReport;
use ss_stats::duplicate_strings::StringDupReport;

/// Severity bucket for an `Advice`. Drives the title color via the existing
/// `OvhdColorHigh` (≥5% of heap) / `OvhdColorMed` CSS classes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AdviceSeverity {
    High,
    Med,
    Low,
}

/// Which panel the advice appears in. Mirrors Java's `displayTitle` prefix
/// convention (`Easy-to-fix problem found:` etc.).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AdviceCategory {
    /// "Easy-to-fix problem found:" — appears in the green "Good news!" panel.
    EasyToFix,
    /// "Preventable problem found:" — All-problems table only.
    Preventable,
    /// "Known problem found:" — All-problems table only.
    Known,
    /// "Potential problem found:" — All-problems table only.
    Potential,
}

impl AdviceCategory {
    pub fn prefix(self) -> &'static str {
        match self {
            AdviceCategory::EasyToFix => "Easy-to-fix problem found:",
            AdviceCategory::Preventable => "Preventable problem found:",
            AdviceCategory::Known => "Known problem found:",
            AdviceCategory::Potential => "Potential problem found:",
        }
    }

    pub fn order(self) -> u8 {
        match self {
            AdviceCategory::EasyToFix => 0,
            AdviceCategory::Preventable => 1,
            AdviceCategory::Known => 2,
            AdviceCategory::Potential => 3,
        }
    }
}

/// One advisor's output. `body_html` is the inner content shown inside the
/// collapsible `<details>` panel.
#[derive(Clone, Debug)]
pub struct Advice {
    pub category: AdviceCategory,
    pub title: String,
    pub severity: AdviceSeverity,
    /// e.g. "Section-18" — used by the "For full details, go to Section N." button.
    pub section_id: &'static str,
    pub overhead_bytes: Option<i64>,
    pub body_html: String,
    /// Stable ordering hint within a category (lower = earlier).
    pub order_hint: u32,
}

impl Advice {
    /// Helper: pick severity based on overhead percentage of total heap.
    pub fn severity_for_pct(pct: f64) -> AdviceSeverity {
        if pct >= 5.0 {
            AdviceSeverity::High
        } else if pct >= 1.0 {
            AdviceSeverity::Med
        } else {
            AdviceSeverity::Low
        }
    }
}

/// Aggregate of all stats reports that advisors may consume. Built once per
/// `generate_advisor_summary` call from `StatsEngine`; advisors read it
/// without further locking.
pub struct ProblemReportSet {
    pub total_bytes: i64,
    pub garbage_bytes: i64,
    pub humongous_bytes: i64,
    pub boxed_overhead_bytes: i64,
    pub string_dup: StringDupReport,
    pub dup_prim: DupArrayReport,
    pub dup_obj: DupArrayReport,
    pub direct_buf: DirectBufferReport,
    pub jvm_flags_raw: String,
    pub gc_type: String,
    pub xmx: i64,
    pub address_size: i32,
    pub java_version: String,
    pub thread_count: usize,
    /// Per-class (name, instance_count, total_bytes) for the top N by bytes.
    pub per_class_top: Vec<(String, u64, i64)>,
}

impl ProblemReportSet {
    pub fn from_stats(stats: &StatsEngine, heap: &HeapDump) -> Self {
        use std::sync::atomic::Ordering::Relaxed;

        let total_bytes = stats.total_shallow_size.load(Relaxed).max(1);
        let garbage_bytes = stats.total_garbage_bytes.load(Relaxed);
        let humongous_bytes: i64 = stats
            .per_class
            .iter()
            .map(|pc| pc.humongous_bytes.load(Relaxed))
            .sum();
        let boxed_overhead_bytes: i64 = stats
            .per_class
            .iter()
            .map(|pc| pc.boxed_number_overhead.load(Relaxed))
            .sum();

        let jvm = stats.jvm_info.get();
        let jvm_flags_raw = jvm.jvm_flags_raw.clone();
        let gc_type = jvm.gc_type.clone();
        let xmx = jvm.xmx;
        let address_size = jvm.address_size;
        let java_version = jvm
            .system_properties
            .iter()
            .find(|(k, _)| k == "java.runtime.version" || k == "java.version")
            .map(|(_, v)| v.clone())
            .unwrap_or_default();

        let thread_count = stats.thread_info.lock().unwrap().len();

        let per_class_top: Vec<(String, u64, i64)> = {
            let mut v: Vec<(String, u64, i64)> = stats
                .per_class
                .iter()
                .map(|pc| {
                    (
                        pc.class_name.clone(),
                        pc.instance_count.load(Relaxed),
                        pc.total_instance_bytes.load(Relaxed),
                    )
                })
                .collect();
            v.sort_by(|a, b| b.2.cmp(&a.2));
            v.truncate(50);
            v
        };

        Self {
            total_bytes,
            garbage_bytes,
            humongous_bytes,
            boxed_overhead_bytes,
            string_dup: stats.string_dup.get_report(),
            dup_prim: stats.dup_prim_arrays.get_report(),
            dup_obj: stats.dup_obj_arrays.get_report(),
            direct_buf: stats.direct_buffers.get_report(),
            jvm_flags_raw,
            gc_type,
            xmx,
            address_size,
            java_version,
            thread_count,
            per_class_top,
        }
    }

    /// Percentage helper.
    pub fn pct(&self, bytes: i64) -> f64 {
        (bytes as f64) * 100.0 / self.total_bytes as f64
    }
}

/// Read-only context passed to each advisor.
pub struct AdvisorContext<'a> {
    pub stats: &'a StatsEngine,
    pub heap: &'a HeapDump,
    pub problems: &'a ProblemReportSet,
}

/// One advisor. Implementations return `Some(Advice)` when the corresponding
/// problem is detected in the heap, or `None` otherwise.
pub trait Advisor: Sync {
    fn name(&self) -> &'static str;
    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice>;
}

/// Registry holding all advisors. Constructed once per report.
pub struct AdvisorRegistry {
    advisors: Vec<Box<dyn Advisor>>,
}

impl AdvisorRegistry {
    /// Build a registry with all known advisors. New advisors are registered
    /// here as they are implemented.
    pub fn new() -> Self {
        let advisors: Vec<Box<dyn Advisor>> = vec![
            // Easy-to-fix (highest priority in the panel)
            Box::new(super::unreachable_direct_buffer::UnreachableDirectBufferAdvisor),
            Box::new(super::old_jdk_string::OldJdkStringWasteAdvisor),
            // Preventable / Known / Potential
            Box::new(super::jvm_flags::JvmFlagsAdvisor),
            Box::new(super::duplicate_strings::DuplicateStringsAdvisor),
            Box::new(super::g1_oom::G1OOMAdvisor),
            Box::new(super::weak_reference::WeakReferenceProliferationAdvisor),
            Box::new(super::optional::OptionalOveruseAdvisor),
            Box::new(super::boxed_number::BoxedNumberAdvisor),
            Box::new(super::object_header::ObjectHeaderOverheadAdvisor),
            Box::new(super::direct_buffer::DirectBufferOffHeapAdvisor),
            Box::new(super::special_direct_buffer::SpecialDirectBufferAdvisor),
            Box::new(super::jackson::JacksonHumongousObjectAdvisor),
            Box::new(super::jetty::JettyHeaderCacheAdvisor),
            Box::new(super::rxjava::RxJavaThreadLeakAdvisor),
            // P3 advisors (heuristic implementations; refine when more data
            // collection is added).
            Box::new(super::netty::NettyPoolChunkAdvisor),
            Box::new(super::ssl::SSLEngineBufferAdvisor),
            Box::new(super::native_thread::NativeThreadOOMAdvisor),
            Box::new(super::exception_stack::ExceptionStackTraceStringAdvisor),
        ];
        Self { advisors }
    }

    /// Run all advisors, returning the non-`None` results in registration order.
    pub fn run_all(&self, ctx: &AdvisorContext) -> Vec<Advice> {
        self.advisors
            .iter()
            .filter_map(|a| a.generate(ctx))
            .collect()
    }
}

impl Default for AdvisorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
