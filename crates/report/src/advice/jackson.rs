//! Advisor: Jackson ObjectMapper humongous objects.
//!
//! Jackson's `ObjectMapper` builds large internal hash tables that often
//! show up as humongous objects under G1 GC. Fires when those tables are
//! detected and humongous bytes are significant.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when humongous bytes ≥ 6% AND a `com.fasterxml.jackson.databind.ObjectMapper`
/// or its main hash table class is present.
pub struct JacksonHumongousObjectAdvisor;

impl Advisor for JacksonHumongousObjectAdvisor {
    fn name(&self) -> &'static str { "JacksonHumongousObjectAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        use std::sync::atomic::Ordering::Relaxed;

        let pct = ctx.problems.pct(ctx.problems.humongous_bytes);
        if pct < 6.0 {
            return None;
        }
        // Look for Jackson's ObjectMapper-related classes.
        let has_jackson = ctx.stats.per_class.iter().any(|pc| {
            pc.class_name.contains("fasterxml/jackson")
                || pc.class_name.contains("jackson/databind")
        });
        if !has_jackson {
            return None;
        }

        let jackson_instances: u64 = ctx.stats.per_class.iter()
            .filter(|pc| pc.class_name.contains("fasterxml/jackson") || pc.class_name.contains("jackson/databind"))
            .map(|pc| pc.instance_count.load(Relaxed))
            .sum();

        let body = format!(
            "We detected <b>{}</b> Jackson (<t-c>com.fasterxml.jackson</t-c>) instances alongside \
             <b>{} ({:.1}%)</b> of humongous objects. Jackson's <t-c>ObjectMapper</t-c> builds \
             large internal hash tables that frequently trigger G1 humongous allocations.<br>\n\
             <br>\n\
             Consider:<br>\n\
             &nbsp;&nbsp;- Reusing a single <code>ObjectMapper</code> instance per application \
             (it's thread-safe after configuration)<br>\n\
             &nbsp;&nbsp;- Avoiding <code>enableDefaultTyping</code> which inflates internal caches<br>\n\
             &nbsp;&nbsp;- Increasing G1 region size (<code>-XX:G1HeapRegionSize=16m</code> or larger)<br>\n",
            jackson_instances,
            crate::table_formatter::format_bytes(ctx.problems.humongous_bytes),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Known,
            title: format!("Jackson humongous objects: {}", crate::table_formatter::format_bytes(ctx.problems.humongous_bytes)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-19",
            overhead_bytes: Some(ctx.problems.humongous_bytes),
            body_html: body,
            order_hint: 900,
        })
    }
}
