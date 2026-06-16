//! Advisor: RxJava cached thread pool leak.
//!
//! RxJava's `RxCachedThreadScheduler` spawns one new worker thread per
//! scheduling request once the pool grows beyond a small core. The thread
//! name pattern is `RxCachedThreadScheduler-N-`. Fires when count ≥ 200.

use super::types::{Advice, AdviceCategory, Advisor, AdvisorContext};

/// Fires when the heap contains ≥ 200 threads whose name starts with
/// `RxCachedThreadScheduler-`.
pub struct RxJavaThreadLeakAdvisor;

impl Advisor for RxJavaThreadLeakAdvisor {
    fn name(&self) -> &'static str { "RxJavaThreadLeakAdvisor" }

    fn generate(&self, ctx: &AdvisorContext) -> Option<Advice> {
        let threads = ctx.stats.thread_info.lock().unwrap();
        let rxc_count = threads.iter()
            .filter(|t| t.name.starts_with("RxCachedThreadScheduler-"))
            .count();
        if rxc_count < 200 {
            return None;
        }

        let body = format!(
            "We found <b>{}</b> threads named <t-c>RxCachedThreadScheduler-N</t-c>. This is \
             strongly indicative of a thread leak in RxJava's scheduler.<br>\n\
             Each thread holds a substantial stack (~512KB-1MB) plus its referenced objects, so \
             leaks of this kind add up quickly.<br>\n\
             <br>\n\
             Common causes:<br>\n\
             &nbsp;&nbsp;- Subscribing on <code>Schedulers.io()</code> or \
             <code>Schedulers.computation()</code> without unsubscribing<br>\n\
             &nbsp;&nbsp;- Using <code>interval</code> / <code>timer</code> operators without \
             disposing the <code>Disposable</code><br>\n\
             &nbsp;&nbsp;- Hot observables that never terminate<br>\n\
             Fix: ensure all subscriptions are tracked via a <code>CompositeDisposable</code> \
             and disposed when the consuming component is destroyed.<br>\n",
            rxc_count,
        );

        Some(Advice {
            category: AdviceCategory::Preventable,
            title: format!("RxJava thread leak: {} RxCachedThreadScheduler threads", rxc_count),
            severity: Advice::severity_for_pct((rxc_count as f64) * 0.05),
            section_id: "Section-22",
            overhead_bytes: Some((rxc_count as i64) * 512 * 1024),
            body_html: body,
            order_hint: 1100,
        })
    }
}
