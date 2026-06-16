//! Advisor: Netty PoolChunk under-utilization (heuristic).
//!
//! Netty's `PoolChunk` allocates fixed 16MB backing arrays per chunk. When
//! the application has many idle event loops, chunks stay allocated but are
//! barely used. Without per-array utilization data, we approximate by
//! counting `io.netty.buffer.PoolChunk` instances.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when `io.netty.buffer.PoolChunk` instances occupy ≥ 2% of heap.
pub struct NettyPoolChunkAdvisor;

impl Advisor for NettyPoolChunkAdvisor {
    fn name(&self) -> &'static str { "NettyPoolChunkAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        use std::sync::atomic::Ordering::Relaxed;

        let mut total_bytes: i64 = 0;
        let mut total_count: u64 = 0;
        for pc in &ctx.stats.per_class {
            if pc.class_name.contains("io/netty/buffer/PoolChunk")
                || pc.class_name.contains("io/netty/util/internal/PoolChunk")
            {
                total_bytes += pc.total_instance_bytes.load(Relaxed);
                total_count += pc.instance_count.load(Relaxed);
            }
        }
        if total_bytes <= 0 {
            return None;
        }
        let pct = ctx.problems.pct(total_bytes);
        if pct < 2.0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> <t-c>io.netty.buffer.PoolChunk</t-c> instances occupying \
             <b>{} ({:.1}%)</b> of heap.<br>\n\
             Each PoolChunk holds a 16MB backing array (default). If many of your event loops are \
             idle, the chunks are still allocated but barely used. Consider:<br>\n\
             &nbsp;&nbsp;- Reducing the number of event loops (typically 2× CPU cores is enough)<br>\n\
             &nbsp;&nbsp;- Setting <code>-Dio.netty.allocator.numDirectArenas</code> lower<br>\n\
             &nbsp;&nbsp;- Switching to <code>UnpooledByteBufAllocator</code> if buffer churn is low<br>\n",
            total_count,
            crate::table_formatter::format_bytes(total_bytes),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Known,
            title: format!("Netty PoolChunk overhead: {}", crate::table_formatter::format_bytes(total_bytes)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-3",
            overhead_bytes: Some(total_bytes),
            body_html: body,
            order_hint: 1200,
        })
    }
}
