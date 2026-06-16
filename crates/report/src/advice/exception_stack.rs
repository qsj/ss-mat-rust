//! Advisor: large exception stack traces as Strings.
//!
//! Java exceptions capture their stack trace as a String via
//! `Throwable.toString()` / `.getStackTrace()`. Long stack traces (e.g.
//! from framework-heavy apps) frequently show up as duplicate Strings
//! that wrap multi-KB stack descriptions. We approximate by checking if
//! String overhead is significant.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when String overhead ≥ 5% AND there's evidence of exception
/// stack traces in the form of `at com.` or `at java.` substrings in
/// the duplicate string samples.
pub struct ExceptionStackTraceStringAdvisor;

impl Advisor for ExceptionStackTraceStringAdvisor {
    fn name(&self) -> &'static str { "ExceptionStackTraceStringAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let dup = &ctx.problems.string_dup;
        let pct = ctx.problems.pct(dup.wasted_bytes);
        if pct < 5.0 {
            return None;
        }

        // Look for stack-trace-shaped content in the top duplicate groups.
        let stacktrace_count = dup.top_groups.iter()
            .filter(|g| g.content.contains("\n\tat ") || g.content.contains("\n    at "))
            .count();
        if stacktrace_count == 0 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> duplicate string groups whose content looks like exception \
             stack traces (containing <code>\\n\\tat ...</code> lines).<br>\n\
             These strings typically originate from <code>Throwable.printStackTrace()</code> or \
             logging frameworks that stringify exceptions. Total duplicate-string overhead is \
             <b>{} ({:.1}%)</b>.<br>\n\
             <br>\n\
             Consider:<br>\n\
             &nbsp;&nbsp;- Not logging full stack traces for expected exceptions<br>\n\
             &nbsp;&nbsp;- Using log filters to suppress noisy exception types<br>\n\
             &nbsp;&nbsp;- Using SLF4J's parameterized logging to avoid string building when \
             the log level is disabled<br>\n",
            stacktrace_count,
            crate::table_formatter::format_bytes(dup.wasted_bytes),
            pct,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("Exception stack traces in Strings: {} groups", stacktrace_count),
            severity: Advice::severity_for_pct(pct),
            section_id: "Section-8",
            overhead_bytes: Some(dup.wasted_bytes / 2),
            body_html: body,
            order_hint: 1500,
        })
    }
}
