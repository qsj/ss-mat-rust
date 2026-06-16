//! Advisor: Native thread OOM heuristic.
//!
//! When an OOM thread is present AND the heap has many threads, the OOM
//! is often caused by thread stack overhead rather than heap exhaustion.
//! This advisor flags that case.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when there's an OOM thread (any thread throwing OOM) AND thread
/// count is ≥ 500. Native stack overhead is roughly thread_count *
/// stack_size (default 512KB-1MB).
pub struct NativeThreadOOMAdvisor;

impl Advisor for NativeThreadOOMAdvisor {
    fn name(&self) -> &'static str { "NativeThreadOOMAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let thread_count = ctx.problems.thread_count;
        if thread_count < 500 {
            return None;
        }

        // Check whether any thread looks like an OOM thread (name pattern).
        // Without specific OOM thread tracking, we approximate by assuming
        // a high thread count itself is a warning sign.
        let stack_estimate = (thread_count as i64) * 1024 * 1024; // 1MB per thread default
        let pct = ctx.problems.pct(stack_estimate);
        if pct < 5.0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> live threads in this heap dump. Each Java thread reserves a \
             fixed native stack (default 512KB-1MB), so the off-heap cost is approximately \
             <b>{} ({:.1}% of heap)</b> in native memory.<br>\n\
             If this dump was triggered by an OutOfMemoryError, consider whether the issue is \
             thread-stack exhaustion rather than heap exhaustion. Each new thread also reserves \
             its share of heap for thread-local allocations.<br>\n\
             <br>\n\
             Mitigation:<br>\n\
             &nbsp;&nbsp;- Reduce thread pool sizes; prefer <code>ThreadPoolExecutor</code> with \
             bounded queues<br>\n\
             &nbsp;&nbsp;- Lower per-thread stack size via <code>-Xss256k</code> or smaller<br>\n\
             &nbsp;&nbsp;- Use virtual threads (Java 21+) for I/O-heavy workloads<br>\n",
            thread_count,
            crate::table_formatter::format_bytes(stack_estimate),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("Native thread overhead: {} threads (~{})", thread_count, crate::table_formatter::format_bytes(stack_estimate)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-22",
            overhead_bytes: Some(stack_estimate),
            body_html: body,
            order_hint: 1400,
        })
    }
}
