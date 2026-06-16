//! Advisor: java.util.Optional overuse.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when `java.util.Optional` field/local usage is ≥ 5% of heap by
/// bytes or by count.
pub struct OptionalOveruseAdvisor;

impl Advisor for OptionalOveruseAdvisor {
    fn name(&self) -> &'static str { "OptionalOveruseAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        use std::sync::atomic::Ordering::Relaxed;

        let mut total_bytes: i64 = 0;
        let mut total_count: u64 = 0;
        for pc in &ctx.stats.per_class {
            if pc.class_name == "java.util.Optional" {
                total_bytes = pc.total_instance_bytes.load(Relaxed);
                total_count = pc.instance_count.load(Relaxed);
                break;
            }
        }
        if total_bytes <= 0 {
            return None;
        }

        let pct_bytes = ctx.problems.pct(total_bytes);
        if pct_bytes < 5.0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> <t-c>java.util.Optional</t-c> instances occupying \
             <b>{} ({:.1}%)</b> of heap.<br>\n\
             <code>Optional</code> was designed as a return type and shouldn't be stored in fields. \
             Each instance adds object header overhead that compounds at scale. Consider using \
             plain nullable references and reserving <code>Optional</code> for method return types.<br>\n",
            total_count,
            crate::table_formatter::format_bytes(total_bytes),
            pct_bytes,
        );

        Some(Advice {
            category: AdviceCategory::Potential,
            title: format!("Optional overuse: {}", crate::table_formatter::format_bytes(total_bytes)),
            severity: Advice::severity_for_pct(pct_bytes),
            section_id: "Section-3",
            overhead_bytes: Some(total_bytes),
            body_html: body,
            order_hint: 500,
        })
    }
}
