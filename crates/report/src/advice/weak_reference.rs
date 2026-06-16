//! Advisor: WeakReference / SoftReference / FinalizerReference proliferation.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when Weak/Soft/PhantomReference instances (excluding Finalizer) occupy
/// ≥ 5% of heap by bytes, OR ≥ 3% by count, OR there are ≥ 50k of them.
pub struct WeakReferenceProliferationAdvisor;

impl Advisor for WeakReferenceProliferationAdvisor {
    fn name(&self) -> &'static str { "WeakReferenceProliferationAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        use std::sync::atomic::Ordering::Relaxed;

        let target_substrings = ["WeakReference", "SoftReference", "PhantomReference", "FinalizerReference", "Cleaner"];
        let mut total_bytes: i64 = 0;
        let mut total_count: u64 = 0;
        for pc in &ctx.stats.per_class {
            let name = &pc.class_name;
            if target_substrings.iter().any(|s| name.contains(s)) {
                total_bytes += pc.total_instance_bytes.load(Relaxed);
                total_count += pc.instance_count.load(Relaxed);
            }
        }
        if total_bytes <= 0 {
            return None;
        }

        let pct_bytes = ctx.problems.pct(total_bytes);
        let pct_count = if ctx.problems.total_bytes > 0 {
            (total_count as f64) * 100.0 / (ctx.problems.total_bytes as f64).max(1.0)
        } else {
            0.0
        };
        let too_many_count = total_count >= 50_000;
        if pct_bytes < 5.0 && pct_count < 3.0 && !too_many_count {
            return None;
        }

        let body = format!(
            "There are <b>{}</b> reference objects (Weak/Soft/Phantom/Finalizer/Cleaner) \
             totaling <b>{} ({:.1}%)</b> of heap.<br>\n\
             Excessive use of reference objects can indicate caching that's too aggressive, \
             or finalization-based cleanup that should be moved to <code>java.lang.ref.Cleaner</code> \
             or explicit <code>close()</code> calls.<br>\n",
            total_count,
            crate::table_formatter::format_bytes(total_bytes),
            pct_bytes,
        );

        Some(Advice {
            category: AdviceCategory::Potential,
            title: format!("Reference object proliferation: {}", crate::table_formatter::format_bytes(total_bytes)),
            severity: Advice::severity_for_pct(pct_bytes),
            section_id: "Section-3",
            overhead_bytes: Some(total_bytes),
            body_html: body,
            order_hint: 400,
        })
    }
}
