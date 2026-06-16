//! Advisor: SSLEngine buffer over-allocation.
//!
//! JSSE's `SSLEngineImpl` allocates large fixed-size buffers (~16KB each
//! for input/output). Many simultaneous TLS connections add up.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when `sun.security.ssl.SSLEngineImpl` (or similar) instances
/// occupy ≥ 2% of heap.
pub struct SSLEngineBufferAdvisor;

impl Advisor for SSLEngineBufferAdvisor {
    fn name(&self) -> &'static str { "SSLEngineBufferAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        use std::sync::atomic::Ordering::Relaxed;

        let mut total_bytes: i64 = 0;
        let mut total_count: u64 = 0;
        for pc in &ctx.stats.per_class {
            let n = pc.class_name.as_str();
            if n.contains("sun/security/ssl/SSLEngineImpl")
                || n.contains("sun/security/ssl/SSLSocketInputRecord")
                || n.contains("sun/security/ssl/SSLSocketOutputRecord")
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
            "We found <b>{}</b> SSL/TLS engine instances (<t-c>SSLEngineImpl</t-c>, \
             input/output recorders) occupying <b>{} ({:.1}%)</b> of heap.<br>\n\
             Each TLS connection holds ~32KB of fixed-size buffers (16KB in, 16KB out). \
             This is normal for many simultaneous connections but adds up at scale. Consider:<br>\n\
             &nbsp;&nbsp;- Using connection pooling to bound live TLS sessions<br>\n\
             &nbsp;&nbsp;- Switching to a TLS implementation that supports buffer pooling<br>\n\
             &nbsp;&nbsp;- Closing idle connections aggressively (set socket read timeouts)<br>\n",
            total_count,
            crate::table_formatter::format_bytes(total_bytes),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Known,
            title: format!("SSL/TLS engine buffer overhead: {}", crate::table_formatter::format_bytes(total_bytes)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-3",
            overhead_bytes: Some(total_bytes),
            body_html: body,
            order_hint: 1300,
        })
    }
}
