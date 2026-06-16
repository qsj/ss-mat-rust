//! Advisor: duplicate strings.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when duplicate string content wastes ≥ 1% of heap.
pub struct DuplicateStringsAdvisor;

impl Advisor for DuplicateStringsAdvisor {
    fn name(&self) -> &'static str { "DuplicateStringsAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let report = &ctx.problems.string_dup;
        if report.wasted_bytes <= 0 {
            return None;
        }
        let pct = ctx.problems.pct(report.wasted_bytes);
        if pct < 1.0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> string objects sharing <b>{}</b> unique contents; \
             deduplicating them would save approximately <b>{} ({:.1}%)</b> of heap.<br>\n\
             Consider interning frequently-repeated strings, or revisiting logging code that emits \
             the same message many times.<br>\n",
            report.total_strings,
            report.unique_strings,
            crate::table_formatter::format_bytes(report.wasted_bytes),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("Duplicate strings wasting {}", crate::table_formatter::format_bytes(report.wasted_bytes)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-8",
            overhead_bytes: Some(report.wasted_bytes),
            body_html: body,
            order_hint: 200,
        })
    }
}
