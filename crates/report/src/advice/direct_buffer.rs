//! Advisor: DirectByteBuffer off-heap memory.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when total DirectByteBuffer off-heap capacity ≥ 5% of heap.
pub struct DirectBufferOffHeapAdvisor;

impl Advisor for DirectBufferOffHeapAdvisor {
    fn name(&self) -> &'static str { "DirectBufferOffHeapAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let cap = ctx.problems.direct_buf.total_capacity;
        if cap <= 0 {
            return None;
        }
        let pct = ctx.problems.pct(cap);
        if pct < 5.0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> <t-c>DirectByteBuffer</t-c> instances with a combined off-heap \
             capacity of <b>{} ({:.1}%)</b> of heap.<br>\n\
             Off-heap memory is not counted toward heap usage but still consumes process RSS. \
             Common causes:<br>\n\
             &nbsp;&nbsp;- NIO channels / Netty ByteBuf pools that grow without bounds<br>\n\
             &nbsp;&nbsp;- Cached buffers that are never explicitly released<br>\n\
             Consider whether all of these buffers are actually in use; release them via \
             <code>sun.misc.Unsafe.invokeCleaner</code> (Java 9+) or use Netty's reference-counting API.<br>\n",
            ctx.problems.direct_buf.total_count,
            crate::table_formatter::format_bytes(cap),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("DirectByteBuffer off-heap usage: {}", crate::table_formatter::format_bytes(cap)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-18",
            overhead_bytes: Some(cap),
            body_html: body,
            order_hint: 800,
        })
    }
}
