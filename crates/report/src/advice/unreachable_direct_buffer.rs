//! Advisor: DirectByteBuffers that are garbage but still hold native memory.
//!
//! This is the "easy-to-fix" advisor from Java. Without per-instance reachability
//! data, we approximate: if total DBB capacity ≥ 5% AND garbage bytes ≥ 20% of heap,
//! there's a strong chance some unreachable DBBs are leaking native memory.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Easy-to-fix: fires when DBB capacity is significant AND garbage ratio is high.
pub struct UnreachableDirectBufferAdvisor;

impl Advisor for UnreachableDirectBufferAdvisor {
    fn name(&self) -> &'static str { "UnreachableDirectBufferAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let cap = ctx.problems.direct_buf.total_capacity;
        if cap <= 0 {
            return None;
        }
        let cap_pct = ctx.problems.pct(cap);
        if cap_pct < 5.0 {
            return None;
        }
        let garbage_pct = ctx.problems.pct(ctx.problems.garbage_bytes);
        if garbage_pct < 20.0 {
            return None;
        }

        let body = format!(
            "We found a potential problem: <t-c>DirectByteBuffer</t-c> objects that are \
             unreachable (garbage) still hold native (off-heap) memory. Total DirectByteBuffer \
             capacity is <b>{} ({:.1}%)</b>, and garbage objects occupy <b>{:.1}%</b> of heap.<br>\n\
             <br>\n\
             To reduce off-heap memory wasted by these DirectByteBuffers, you may need to:<br>\n\
             &nbsp;&nbsp;- Reduce the heap size (via <code>-Xmx</code>) so that GC happens more \
             frequently, freeing DirectByteBuffers sooner<br>\n\
             &nbsp;&nbsp;- Explicitly release buffers when done (Netty: <code>ByteBuf.release()</code>, \
             JDK 9+: <code>sun.misc.Unsafe.invokeCleaner(bb)</code>)<br>\n\
             &nbsp;&nbsp;- Reduce the number of DirectByteBuffers allocated<br>\n",
            crate::table_formatter::format_bytes(cap),
            cap_pct,
            garbage_pct,
        );

        Some(Advice {
            category: AdviceCategory::EasyToFix,
            title: format!("Unreachable DirectByteBuffers may be leaking native memory"),
            severity: Advice::severity_for_pct(cap_pct),
            section_id: "Section-18",
            overhead_bytes: Some(cap),
            body_html: body,
            order_hint: 60,
        })
    }
}
