//! Advisor: "special" DirectByteBuffers (Netty/buffer-pool subclasses).
//!
//! Detects DirectByteBuffer subclasses from common NIO frameworks (Netty,
//! etc.) so the user can investigate them as a separate class of buffers.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when ≥ 1 DirectByteBuffer subclass (not the base
/// `java.nio.DirectByteBuffer`) has capacity ≥ 1% of heap.
pub struct SpecialDirectBufferAdvisor;

impl Advisor for SpecialDirectBufferAdvisor {
    fn name(&self) -> &'static str { "SpecialDirectBufferAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let mut special_total: i64 = 0;
        let mut special_count: usize = 0;
        let mut top_entry: Option<(&str, i64)> = None;
        for entry in &ctx.problems.direct_buf.per_class {
            // The base DirectByteBuffer class itself is not "special".
            if entry.class_name == "java.nio.DirectByteBuffer" {
                continue;
            }
            special_total += entry.total_capacity;
            special_count += entry.count;
            match top_entry {
                None => top_entry = Some((entry.class_name.as_str(), entry.total_capacity)),
                Some((_, b)) if entry.total_capacity > b => {
                    top_entry = Some((entry.class_name.as_str(), entry.total_capacity))
                }
                _ => {}
            }
        }
        if special_total <= 0 {
            return None;
        }
        let pct = ctx.problems.pct(special_total);
        if pct < 1.0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> DirectByteBuffer subclass instances totaling \
             <b>{} ({:.1}%)</b> of off-heap-equivalent capacity.<br>\n\
             These are typically Netty <t-c>PooledByteBuf</t-c> instances or framework-specific \
             buffer wrappers. They may indicate a pool that has grown beyond the application's \
             actual needs, or a buffer leak in the framework.<br>\n\
             Top subclass: <t-c>{}</t-c>.<br>\n",
            special_count,
            crate::table_formatter::format_bytes(special_total),
            pct,
            top_entry.map(|(n, _)| n.replace('/', ".")).unwrap_or_else(|| "n/a".into()),
        );

        Some(Advice {
            category: AdviceCategory::Known,
            title: format!("Special DirectByteBuffers: {}", crate::table_formatter::format_bytes(special_total)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-18",
            overhead_bytes: Some(special_total),
            body_html: body,
            order_hint: 810,
        })
    }
}
