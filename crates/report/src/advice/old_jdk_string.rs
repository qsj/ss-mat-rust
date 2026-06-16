//! Advisor: String duplication savings only achievable on JDK ≥ 9.
//!
//! JDK 9+ stores `String` as `byte[]` with a per-string coder flag, which
//! uses roughly half the backing storage of JDK 8's `char[]` for Latin-1
//! content. Upgrading alone can save ≥ 4% on Latin-1-heavy heaps.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when the JVM reports a version < 9 (or version unknown) AND
/// duplicate strings or `char[]` overhead is significant.
pub struct OldJdkStringWasteAdvisor;

impl Advisor for OldJdkStringWasteAdvisor {
    fn name(&self) -> &'static str { "OldJdkStringWasteAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        // Parse major version from java.version string.
        let major = parse_major_version(&ctx.problems.java_version);
        // If unknown (0), skip; if >= 9, also skip.
        if major == 0 || major >= 9 {
            return None;
        }

        let dup = ctx.problems.string_dup.wasted_bytes;
        let dup_pct = ctx.problems.pct(dup);
        if dup_pct < 4.0 {
            return None;
        }

        let body = format!(
            "You're running on Java <b>{}</b>, where <t-c>java.lang.String</t-c> stores its \
             content as <t-c>char[]</t-c> (2 bytes per character). Java 9+ introduced compact \
             strings, storing Latin-1 content as <t-c>byte[]</t-c> (1 byte per character).<br>\n\
             We see <b>{} ({:.1}%)</b> of duplicate-string overhead. After upgrading to Java 9+ \
             and enabling <code>-XX:+CompactStrings</code> (default), you'd save roughly half \
             of all string backing storage that's Latin-1, plus benefit from G1 string \
             deduplication (<code>-XX:+UseStringDeduplication</code>).<br>\n",
            major,
            crate::table_formatter::format_bytes(dup),
            dup_pct,
        );

        Some(Advice {
            category: AdviceCategory::EasyToFix,
            title: format!("Upgrading from Java {} would compact strings", major),
            severity: Advice::severity_for_pct(dup_pct / 2.0),
            section_id: "Section-8",
            overhead_bytes: Some(dup / 2),
            body_html: body,
            order_hint: 70,
        })
    }
}

fn parse_major_version(v: &str) -> u32 {
    // Strip vendor prefixes (e.g., "1.8.0_301" → 8; "11.0.12" → 11).
    let trimmed = v.trim();
    if trimmed.is_empty() {
        return 0;
    }
    let first_num_start = trimmed.find(|c: char| c.is_ascii_digit()).unwrap_or(0);
    let rest = &trimmed[first_num_start..];
    // Handle "1.X" form for Java ≤ 8
    let parts: Vec<&str> = rest.split(|c: char| !c.is_ascii_digit()).collect();
    if parts.is_empty() {
        return 0;
    }
    let n1: u32 = parts[0].parse().unwrap_or(0);
    if n1 == 1 && parts.len() > 1 {
        parts[1].parse().unwrap_or(0)
    } else {
        n1
    }
}
