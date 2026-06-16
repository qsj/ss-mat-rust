//! Advisor: G1 GC + OOM from humongous objects.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when the heap is using G1 GC and humongous objects occupy ≥ 6%
/// of heap (likely cause of OOM).
pub struct G1OOMAdvisor;

impl Advisor for G1OOMAdvisor {
    fn name(&self) -> &'static str { "G1OOMAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        if !ctx.problems.gc_type.to_ascii_uppercase().contains("G1") {
            return None;
        }
        if ctx.problems.humongous_bytes <= 0 {
            return None;
        }
        let pct = ctx.problems.pct(ctx.problems.humongous_bytes);
        if pct < 6.0 {
            return None;
        }

        let body = format!(
            "Your JVM appears to be using the <b>G1</b> garbage collector, and we found \
             <b>{} ({:.1}%)</b> of memory occupied by humongous objects (objects larger than \
             half a G1 region).<br>\n\
             G1 treats these objects specially and they frequently cause premature OOM. Consider:<br>\n\
             &nbsp;&nbsp;- Increasing G1 region size via <code>-XX:G1HeapRegionSize</code><br>\n\
             &nbsp;&nbsp;- Reducing the size of large allocations (caches, buffers)<br>\n\
             &nbsp;&nbsp;- Switching to a collector that handles large objects better<br>\n",
            crate::table_formatter::format_bytes(ctx.problems.humongous_bytes),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("G1 GC with {} of humongous objects", crate::table_formatter::format_bytes(ctx.problems.humongous_bytes)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-19",
            overhead_bytes: Some(ctx.problems.humongous_bytes),
            body_html: body,
            order_hint: 300,
        })
    }
}
