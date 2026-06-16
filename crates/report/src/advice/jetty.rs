//! Advisor: Jetty header cache over-allocation.
//!
//! Jetty's `HttpField` cache pre-allocates large Object[] arrays for header
//! lookups. Fires when those arrays waste ≥ 2% of heap.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when Jetty classes are present AND bad-object-array overhead is ≥ 2%.
/// Without per-class array data, we use total instance bytes for
/// `org.eclipse.jetty` classes as a proxy.
pub struct JettyHeaderCacheAdvisor;

impl Advisor for JettyHeaderCacheAdvisor {
    fn name(&self) -> &'static str { "JettyHeaderCacheAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        use std::sync::atomic::Ordering::Relaxed;

        let mut jetty_bytes: i64 = 0;
        let mut jetty_count: u64 = 0;
        for pc in &ctx.stats.per_class {
            if pc.class_name.contains("org/eclipse/jetty") {
                jetty_bytes += pc.total_instance_bytes.load(Relaxed);
                jetty_count += pc.instance_count.load(Relaxed);
            }
        }
        if jetty_bytes <= 0 {
            return None;
        }
        let pct = ctx.problems.pct(jetty_bytes);
        if pct < 2.0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> Jetty (<t-c>org.eclipse.jetty</t-c>) instances occupying \
             <b>{} ({:.1}%)</b> of heap.<br>\n\
             Jetty's <t-c>HttpConfiguration</t-c> and header caches are common sources of overhead, \
             particularly the pre-allocated Object[] arrays used for header lookups. Consider:<br>\n\
             &nbsp;&nbsp;- Reducing <code>outputBufferSize</code> / <code>requestBufferSize</code> in \
             <code>HttpConfiguration</code><br>\n\
             &nbsp;&nbsp;- Reviewing custom header cache size if you have one<br>\n",
            jetty_count,
            crate::table_formatter::format_bytes(jetty_bytes),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("Jetty overhead: {}", crate::table_formatter::format_bytes(jetty_bytes)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-3",
            overhead_bytes: Some(jetty_bytes),
            body_html: body,
            order_hint: 1000,
        })
    }
}
