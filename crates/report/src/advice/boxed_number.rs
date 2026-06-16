//! Advisor: boxed number overhead.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when boxed number wrapper objects (Integer, Long, Double, ...) waste
/// ≥ 1% of heap on object header overhead.
pub struct BoxedNumberAdvisor;

impl Advisor for BoxedNumberAdvisor {
    fn name(&self) -> &'static str { "BoxedNumberAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let overhead = ctx.problems.boxed_overhead_bytes;
        if overhead <= 0 {
            return None;
        }
        let pct = ctx.problems.pct(overhead);
        if pct < 1.0 {
            return None;
        }

        let body = format!(
            "Boxed numbers (<t-c>Integer</t-c>, <t-c>Long</t-c>, <t-c>Double</t-c>, ...) are \
             causing approximately <b>{} ({:.1}%)</b> of overhead due to object headers (typically \
             16 bytes per wrapper vs. 4-8 bytes for the underlying primitive).<br>\n\
             Consider:<br>\n\
             &nbsp;&nbsp;- Using primitive types (<code>int</code>, <code>long</code>) in fields \
             where possible<br>\n\
             &nbsp;&nbsp;- Using <code>EnumMap</code> / specialized primitive collections \
             (e.g. fastutil, Eclipse Collections)<br>\n\
             &nbsp;&nbsp;- Caching hot values via <code>Integer.valueOf</code> / <code>Boolean.valueOf</code><br>\n",
            crate::table_formatter::format_bytes(overhead),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("Boxed number overhead: {}", crate::table_formatter::format_bytes(overhead)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-12",
            overhead_bytes: Some(overhead),
            body_html: body,
            order_hint: 600,
        })
    }
}
