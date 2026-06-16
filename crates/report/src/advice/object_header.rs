//! Advisor: per-object fixed overhead (headers) for collection-like classes.
//!
//! Approximates Java's `ObjectHeaderOverheadAdvisor`. Java keys off
//! `BoxedNumberOverheadSummary` for known collection classes; here we sum
//! shallow bytes for the standard JDK collection classes and compare to a
//! notional "payload" estimate.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Known JDK collection/map classes whose per-instance header is the
/// dominant cost when contents are small.
const COLLECTION_CLASSES: &[&str] = &[
    "java/util/HashMap",
    "java/util/LinkedHashMap",
    "java/util/TreeMap",
    "java/util/ConcurrentHashMap",
    "java/util/ArrayList",
    "java/util/LinkedList",
    "java/util/HashSet",
    "java/util/TreeSet",
    "java/util/LinkedHashSet",
];

/// Fires when collection/map instances collectively waste ≥ 2% of heap on
/// per-object headers.
pub struct ObjectHeaderOverheadAdvisor;

impl Advisor for ObjectHeaderOverheadAdvisor {
    fn name(&self) -> &'static str { "ObjectHeaderOverheadAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        use std::sync::atomic::Ordering::Relaxed;

        let mut total_bytes: i64 = 0;
        let mut total_count: u64 = 0;
        let mut top_name: Option<String> = None;
        let mut top_bytes: i64 = 0;
        for pc in &ctx.stats.per_class {
            if COLLECTION_CLASSES.iter().any(|c| pc.class_name.as_str() == *c) {
                let bytes = pc.total_instance_bytes.load(Relaxed);
                let count = pc.instance_count.load(Relaxed);
                total_bytes += bytes;
                total_count += count;
                if bytes > top_bytes {
                    top_bytes = bytes;
                    top_name = Some(pc.class_name.replace('/', "."));
                }
            }
        }
        if total_bytes <= 0 || total_count == 0 {
            return None;
        }
        // Header is ~16 bytes/instance; "overhead" = header * count.
        let header_overhead = (total_count as i64) * 16;
        let pct = ctx.problems.pct(header_overhead);
        if pct < 2.0 {
            return None;
        }

        let body = format!(
            "There are <b>{}</b> collection/map instances (<t-c>HashMap</t-c>, \
             <t-c>ArrayList</t-c>, <t-c>ConcurrentHashMap</t-c>, ...) occupying \
             <b>{} ({:.1}%)</b> of heap; per-instance object headers account for roughly \
             <b>{} ({:.1}%)</b> of that.<br>\n\
             Many small collections are a common source of overhead. Consider consolidating \
             maps/lists, or using primitive-specialized collections (fastutil, Eclipse Collections) \
             to avoid both per-instance headers and per-element boxing.<br>\n\
             Top class: <t-c>{}</t-c>.<br>\n",
            total_count,
            crate::table_formatter::format_bytes(total_bytes),
            ctx.problems.pct(total_bytes),
            crate::table_formatter::format_bytes(header_overhead),
            pct,
            top_name.unwrap_or_else(|| "n/a".into()),
        );

        Some(Advice {
            category: AdviceCategory::Potential,
            title: format!("Collection object-header overhead: {}", crate::table_formatter::format_bytes(header_overhead)),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-9",
            overhead_bytes: Some(header_overhead),
            body_html: body,
            order_hint: 700,
        })
    }
}
