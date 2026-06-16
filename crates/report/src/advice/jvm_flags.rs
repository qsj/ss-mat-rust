//! Advisor: JVM flags not provided.

use super::types::{Advice, AdviceCategory, AdviceSeverity, Advisor, AdvisorContext};

/// Detects the case where JVM flags (`-Xmx`, `-XX:...`) are not attached to
/// the heap dump. Without them, several analyses are less precise.
pub struct JvmFlagsAdvisor;

impl Advisor for JvmFlagsAdvisor {
    fn name(&self) -> &'static str { "JvmFlagsAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        if !ctx.problems.jvm_flags_raw.is_empty() {
            return None;
        }

        let body = "\
We found a potential problem: <t-c>JVM flags</t-c> are not provided.<br>\n<br>\n\
When the JVM flags (java command line arguments) for the app that generated this heap dump\n\
are not provided, several types of analysis can be much less precise.<br>\n<br>\n\
To include them, add the following code early in your app:<br>\n<br>\n\
<code>&nbsp;&nbsp;import java.lang.management.ManagementFactory;<br>\n\
&nbsp;&nbsp;List&lt;String&gt; jvmArgs = ManagementFactory.getRuntimeMXBean().getInputArguments();<br>\n\
&nbsp;&nbsp;System.getProperties().put(\"java.vm.inputarguments\", String.join(\" \", jvmArgs));</code><br>\n\
".to_string();

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: "JVM flags are not provided".into(),
            severity: AdviceSeverity::Med,
            section_id: "Section-24",
            overhead_bytes: None,
            body_html: body,
            order_hint: 100,
        })
    }
}
