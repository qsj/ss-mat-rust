use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::path::PathBuf;

use ss_heap_model::heap_dump::{FieldValue, HeapDump, LocalVarDetail};
use ss_stats::engine::StatsEngine;
use ss_stats::per_class::PerClassStats;

use crate::advice::{
    Advice, AdviceCategory, AdviceSeverity, AdvisorContext, AdvisorRegistry, ProblemReportSet,
};
use crate::html_builder::HtmlStringBuilder;
use crate::ref_chain_render::{
    render_class_ref_chains, render_humongous_ref_chains, render_tree_node,
    ProblemObj, RefChainRenderInput, RefChainRenderer,
};
use crate::table_formatter::{
    bigger_font, bold, calc_percent, format_bytes, format_count,
    format_percent, format_section_header, format_with_percent, severity_high, severity_low,
    severity_med,
};

/// Java's `JavaClassInfo.isObjectArray()`: any array whose element type is not
/// a primitive. Multi-dimensional primitive arrays (`[[I`, `[[B`) are primitive
/// arrays in Java and must be excluded from Section 10's total.
fn is_object_array_class_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes[0] != b'[' {
        return false;
    }
    let mut i = 0usize;
    while i < bytes.len() && bytes[i] == b'[' {
        i += 1;
    }
    i < bytes.len() && bytes[i] == b'L'
}

/// Java's ContainerRegistry class list (exact-class match). Used for the
/// "Total collections" denominator, which is the sum of instanceCount for all
/// classes registered as containers.
fn is_collection_class_name_for_report(name: &str) -> bool {
    matches!(name,
        "java/util/ArrayList" | "java/util/Vector" | "java/util/Stack"
        | "java/util/HashMap" | "java/util/LinkedHashMap" | "java/util/Hashtable"
        | "java/util/Properties" | "java/util/WeakHashMap"
        | "java/lang/ThreadLocal$ThreadLocalMap"
        | "java/util/concurrent/ConcurrentHashMap"
        | "java/util/TreeMap" | "java/util/HashSet" | "java/util/LinkedHashSet"
        | "java/util/TreeSet" | "java/util/LinkedList"
        | "java/util/IdentityHashMap"
        | "java/util/concurrent/ConcurrentSkipListMap"
        | "java/util/concurrent/LinkedBlockingQueue"
        | "java/util/concurrent/LinkedBlockingDeque")
}

/// Format a field value as an HTML string matching Java report style.
fn format_field_value(val: &FieldValue) -> String {
    match val {
        FieldValue::Reference(None) => "<t-s>null</t-s>".to_string(),
        FieldValue::Reference(Some(class_name)) => {
            format!("<t-c>{}</t-c>", class_name)
        }
        FieldValue::Boolean(v) => format!("<t-s>{}</t-s>", v),
        FieldValue::Byte(v) => format!("<t-s>{}</t-s>", v),
        FieldValue::Char(v) => {
            let ch = char::from_u32(*v as u32).unwrap_or('?');
            if *v >= 0x20 && *v < 0x7f {
                format!("<t-s>'{}'</t-s>", ch)
            } else {
                format!("<t-s>\\u{:04x}</t-s>", v)
            }
        }
        FieldValue::Short(v) => format!("<t-s>{}</t-s>", v),
        FieldValue::Int(v) => format!("<t-s>{}</t-s>", v),
        FieldValue::Long(v) => format!("<t-s>{}</t-s>", v),
        FieldValue::Float(bits) => {
            let f = f32::from_bits(*bits);
            format!("<t-s>{}</t-s>", f)
        }
        FieldValue::Double(bits) => {
            let d = f64::from_bits(*bits);
            format!("<t-s>{}</t-s>", d)
        }
    }
}

/// Format a LocalVarDetail as HTML matching Java report style.
/// Instance: `<rc_tab2><t-c>ClassName</t-c>(<t-f>field</t-f> : value, ...)`
/// Object array: `<rc_tab2><t-c>ClassName</t-c>[N]{elem0, elem1, ...}`
/// Primitive array: `<rc_tab2><t-c>ClassName</t-c>[N]{...}`
fn format_local_var_detail(detail: &LocalVarDetail) -> String {
    let mut s = String::with_capacity(1024);

    if let Some(display) = &detail.prim_array_display {
        // Primitive array
        s.push_str(&format!("<rc_tab2><t-c>{}</t-c>{}", detail.class_name, html_escape(&display)));
    } else if let Some((elem_class, elements)) = &detail.array_elements {
        // Object array
        let len = elements.len();
        s.push_str(&format!("<rc_tab2><t-c>{}</t-c>[{}]{{", detail.class_name, len));
        for (i, elem) in elements.iter().enumerate() {
            if i > 0 { s.push_str(", "); }
            match elem {
                Some(name) => s.push_str(&format!("<t-c>{}</t-c>", name)),
                None => s.push_str("<t-s>null</t-s>"),
            }
        }
        s.push('}');
    } else if !detail.fields.is_empty() {
        // Instance with fields
        s.push_str(&format!("<rc_tab2><t-c>{}</t-c>(", detail.class_name));
        for (i, (name, val)) in detail.fields.iter().enumerate() {
            if i > 0 { s.push_str(", "); }
            s.push_str(&format!("<t-f>{}</t-f> : {}", name, format_field_value(val)));
        }
        s.push(')');
    } else {
        // Fallback: just class name and address
        s.push_str(&format!("<rc_tab2><t-c>{}</t-c>@{:x}", detail.class_name, detail.address));
    }

    s
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Main report generator matching Java JXRay report format with 24 sections + advisor.
pub struct MainReportGenerator {
    heap: Arc<HeapDump>,
    stats: Arc<StatsEngine>,
    section_count: usize,
    dump_file_name: String,
    /// Path to the HPROF file, used to re-parse field values for `<t-s>`
    /// scalar rendering inside reference chains. `None` when not available
    /// (value rendering is then skipped).
    hprof_path: Option<PathBuf>,
}

impl MainReportGenerator {
    pub fn new(heap: Arc<HeapDump>, stats: Arc<StatsEngine>, dump_file_name: String) -> Self {
        Self {
            heap,
            stats,
            section_count: 0,
            dump_file_name,
            hprof_path: None,
        }
    }

    /// Set the HPROF file path so the generator can re-parse field values at
    /// render time (enables `<t-s>` scalar tags inside reference chains).
    pub fn with_hprof_path(mut self, hprof_path: impl Into<PathBuf>) -> Self {
        self.hprof_path = Some(hprof_path.into());
        self
    }

    fn next_section_id(&mut self) -> String {
        self.section_count += 1;
        format!("Section-{}", self.section_count)
    }

    fn section_numbered_title(&self) -> String {
        format!("{}. ", self.section_count)
    }

    /// Render "Typical reference paths" for a class by looking up the existing
    /// per-class `ref_chains` data. Returns empty HTML if no chains are
    /// available for this class. Used by problem sections to expose the
    /// common paths under which instances of the class live, without
    /// requiring per-instance tracking in collectors.
    fn render_class_ref_chain_panel(&self, class_name: &str, total_bytes: i64) -> String {
        // Resolve class_idx by name. Class names in stats use '/' package
        // separators; the caller passes either form.
        let lookup = class_name.replace('.', "/");
        let class_idx = match self.heap.class_by_name.get(lookup.as_str()) {
            Some(&idx) => idx,
            None => return String::new(),
        };
        let ref_chains = self.stats.ref_chains.lock().unwrap();
        let chains = match ref_chains.chains.get(&class_idx) {
            Some(c) => c.clone(),
            None => return String::new(),
        };
        if chains.expensive_fields.is_empty() && chains.full_chains.is_empty() {
            return String::new();
        }
        let mut html = String::with_capacity(2048);
        html.push_str(&format!(
            "<br><details><summary>Typical reference paths for <t-c>{}</t-c></summary><br>\n",
            class_name.replace('/', ".")
        ));
        html.push_str(&render_class_ref_chains(&chains, total_bytes));
        html.push_str("</details>\n");
        html
    }

    /// Render per-instance reference chains for a list of problem object IDs.
    /// Used by problem sections (8/14/15/18/...) where collectors have
    /// captured sample_obj_ids of problem instances. Returns HTML string
    /// directly (caller appends to `html`).
    fn render_instance_ref_chains(
        &self,
        html: &mut HtmlStringBuilder,
        title: &str,
        size_bytes: i64,
        total_bytes: i64,
        sample_obj_ids: &[u64],
        sec_sub_title: Option<&str>,
    ) {
        let mut buf = String::new();
        self.render_instance_ref_chains_into(
            &mut buf, title, size_bytes, total_bytes, sample_obj_ids, sec_sub_title,
        );
        if !buf.is_empty() {
            html.append_raw(&buf);
        }
    }

    /// Core implementation writing into a `String`. Returns without writing
    /// when there is nothing to render (no samples / no matching objects).
    fn render_instance_ref_chains_into(
        &self,
        out: &mut String,
        title: &str,
        size_bytes: i64,
        total_bytes: i64,
        sample_obj_ids: &[u64],
        sec_sub_title: Option<&str>,
    ) {
        if sample_obj_ids.is_empty() {
            return;
        }

        let guard = self.stats.traversal_data.lock().unwrap();
        let data = match guard.as_ref() {
            Some(d) => d,
            None => return,
        };

        // Build obj_id → obj_idx lookup map for the samples.
        // Linear scan over a slice of N samples × M objects is acceptable
        // for N≤20. For larger N, build a HashSet first.
        let sample_set: std::collections::HashSet<u64> = sample_obj_ids.iter().cloned().collect();
        let mut id_to_idx: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        for &id in &sample_set {
            if let Some(rank) = data.rank_of_id(id) {
                id_to_idx.insert(id, rank);
            }
        }

        if id_to_idx.is_empty() {
            return;
        }

        // Borrowed, not cloned: the root set has one entry per GC root and the
        // renderer only needs to look members up.
        let gc_root_set = &data.gc_roots;

        // Preserve caller's order; map each obj_id to ProblemObj.
        let problem_objs: Vec<ProblemObj> = sample_obj_ids.iter()
            .filter_map(|id| {
                id_to_idx.get(id).and_then(|&idx| {
                    data.object_at(idx).map(|obj| {
                        let class_name = self.get_class_display(obj.class_idx);
                        ProblemObj {
                            obj_idx: idx,
                            shallow_bytes: 0,
                            display: format!("<t-c>{}</t-c>@0x{:x}", class_name, obj.obj_id),
                        }
                    })
                })
            })
            .take(20)
            .collect();

        if problem_objs.is_empty() {
            return;
        }

        // Resolve per-object field VALUES for the problem objects and their
        // chain ancestors, so primitive fields render as `<t-s>` tags.
        // Walk parent[] from each problem object to collect ancestor obj_ids.
        let mut chain_addrs: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for prob in &problem_objs {
            let mut cur = prob.obj_idx;
            for _ in 0..60 {
                if let Some(obj) = data.object_at(cur) {
                    chain_addrs.insert(obj.obj_id);
                }
                let (p, _) = data.parent.parent_of(cur);
                if p == u32::MAX {
                    break;
                }
                cur = p;
            }
        }
        let live_before = ss_support::mem_trace::live_bytes();
        let resolved = self.hprof_path.as_ref().and_then(|path| {
            if chain_addrs.is_empty() {
                None
            } else {
                Some(ss_traversal::resolve_field_values(&self.heap, path, &chain_addrs))
            }
        });
        if let Some(_rv) = &resolved {
            let held = ss_support::mem_trace::live_bytes().saturating_sub(live_before);
            ss_support::mem_trace::trace(&format!(
                "rv-held addrs={} held={}KB",
                chain_addrs.len(),
                held / 1024
            ));
        }

        let renderer = crate::ref_chain_render::RefChainRenderer::new(
            &self.heap,
            data,
            &data.field_pool,
            gc_root_set,
        );
        let renderer = match &resolved {
            Some(rv) => renderer.with_resolved_values(rv),
            None => renderer,
        };

        let input = RefChainRenderInput {
            title: title.to_string(),
            size_bytes,
            total_bytes,
            problem_objs: &problem_objs,
            max_objs_to_render: 20,
            sec_sub_title: sec_sub_title.map(|s| s.to_string()),
        };
        out.push_str(&renderer.render_problem_list(input));
    }

    /// Generate the full HTML report.
    pub fn generate(&mut self) -> String {
        self.try_generate().expect("read report result stores")
    }

    pub fn try_generate(&mut self) -> std::io::Result<String> {
        let mut html = HtmlStringBuilder::new();

        // Placeholder for advisor summary (prepended later)
        html.append_raw("<!-- ADVISOR_PLACEHOLDER -->");

        // --- Section 1: Top-Level Stats ---
        self.section_top_level_stats(&mut html);
        ss_support::mem_trace::trace("s01-top-stats");

        // --- Section 2: Thread throwing OutOfMemoryError ---
        self.section_oom_thread(&mut html);
        ss_support::mem_trace::trace("s02-oom-thread");

        // --- Section 3: Where Memory Goes, by Class ---
        self.section_memory_by_class(&mut html);
        ss_support::mem_trace::trace("s03-by-class");

        // --- Section 4: Where Memory Goes, by GC Root ---
        self.section_memory_by_gc_root(&mut html);
        ss_support::mem_trace::trace("s04-by-gc-root");

        // --- Section 5: Live vs Garbage Objects ---
        self.section_live_vs_garbage(&mut html);
        ss_support::mem_trace::trace("s05-live-vs-garbage");

        // --- Section 6: Fixed per-object overhead ---
        self.section_fixed_overhead(&mut html);
        ss_support::mem_trace::trace("s06-fixed-overhead");

        // --- Section 7: Objects Awaiting Finalization ---
        self.section_finalization(&mut html);
        ss_support::mem_trace::trace("s07-finalization");

        // --- Section 8: Duplicate Strings ---
        self.section_duplicate_strings(&mut html);
        ss_support::mem_trace::trace("s08-dup-strings");

        // --- Section 9: Bad Collections ---
        self.section_bad_collections(&mut html)?;
        ss_support::mem_trace::trace("s09-bad-collections");

        // --- Section 10: Bad Object Arrays ---
        self.section_bad_obj_arrays(&mut html)?;
        ss_support::mem_trace::trace("s10-bad-obj-arrays");

        // --- Section 11: Bad Primitive Arrays ---
        self.section_bad_prim_arrays(&mut html)?;
        ss_support::mem_trace::trace("s11-bad-prim-arrays");

        // --- Section 12: Boxed Numbers ---
        self.section_boxed_numbers(&mut html);
        ss_support::mem_trace::trace("s12-boxed-numbers");

        // --- Section 13: Duplicate Objects ---
        self.section_duplicate_objects(&mut html);
        ss_support::mem_trace::trace("s13-dup-objects");

        // --- Section 14: Duplicate Primitive Arrays ---
        self.section_dup_prim_arrays(&mut html);
        ss_support::mem_trace::trace("s14-dup-prim-arrays");

        // --- Section 15: Duplicate Object Arrays ---
        self.section_dup_obj_arrays(&mut html);
        ss_support::mem_trace::trace("s15-dup-obj-arrays");

        // --- Section 16: Duplicate Lists ---
        self.section_dup_lists(&mut html);
        ss_support::mem_trace::trace("s16-dup-lists");

        // --- Section 17: WeakHashMaps ---
        self.section_weak_hashmaps(&mut html);
        ss_support::mem_trace::trace("s17-weak-hashmaps");

        // --- Section 18: Off-heap DirectByteBuffer memory ---
        self.section_direct_byte_buffers(&mut html);
        ss_support::mem_trace::trace("s18-direct-buffers");

        // --- Section 19: Humongous objects ---
        self.section_humongous_objects(&mut html);
        ss_support::mem_trace::trace("s19-humongous");

        // --- Section 20: Heap Size Configuration ---
        self.section_heap_config(&mut html);
        ss_support::mem_trace::trace("s20-heap-config");

        // --- Section 21: Very Long Reference Chains ---
        self.section_long_ref_chains(&mut html);
        ss_support::mem_trace::trace("s21-long-chains");

        // --- Section 22: Thread stacks ---
        self.section_thread_stacks(&mut html);
        ss_support::mem_trace::trace("s22-thread-stacks");

        // --- Section 23: System Properties ---
        self.section_system_properties(&mut html);
        ss_support::mem_trace::trace("s23-sys-props");

        // --- Section 24: JVM Flags ---
        self.section_jvm_flags(&mut html);
        ss_support::mem_trace::trace("s24-jvm-flags");

        // Close last section
        html.append_details_close_tag();

        // Generate advisor summary and prepend
        let advisor_html = self.generate_advisor_summary();
        let body = html.build().replace("<!-- ADVISOR_PLACEHOLDER -->", &advisor_html);

        // Wrap with full HTML document
        let mut doc = String::with_capacity(body.len() + 8192);
        doc.push_str("<html>\n<head>\n<meta charset=\"UTF-8\">\n<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n<title>");
        doc.push_str(&format!("SS-Mat report for {}", self.dump_file_name));
        doc.push_str("</title>\n<style>\n");
        doc.push_str(include_str!("../css_report.txt"));
        doc.push_str("\n</style>\n<script>\n");
        doc.push_str(include_str!("../js_report.txt"));
        // Append encoded-string dedup decoder + the populated ENCODED_STRINGS
        // array (strings collected during reference-chain rendering). Matches
        // Java's HtmlStringBuilder head block.
        doc.push_str("\nvar ENCODED_STRINGS=[");
        doc.push_str(&crate::encoded_strings::take_array_js());
        doc.push_str("];\n");
        doc.push_str(r#"var ESP="|*/",ESS="|",ESC='c',ESSC='s';
function replEncStrs(id){var el=document.getElementById('sd-'+id);rplRec(el)}
function rplRec(el){var ch=el.children;for(var i=0;i<ch.length;i++){rplEl(ch[i]);rplRec(ch[i])}}
function rplEl(el){var rt='',si=0,t=el.innerHTML,pi=t.indexOf(ESP);while(pi!=-1){rt+=t.substring(si,pi);var ni=pi+3,st=0,nc=t.charAt(ni);if(nc==ESC)st=1;else if(nc==ESSC)st=2;if(st==0){si=ni;rt+=t.substring(pi,ni);pi=t.indexOf(ESP,si);continue}var nei=t.indexOf(ESS,ni+1);if(nei!=-1&&nei-ni<=7){var ns=t.substring(ni+1,nei),n=parseInt(ns);if(!isNaN(n)&&n>=0&&n<=ENCODED_STRINGS.length){var s=ENCODED_STRINGS[n];rt+=(st==1?'<t-c>'+s+'</t-c>':'<t-s>'+s+'</t-s>')}else{rt+=t.substring(pi,nei+1)}si=nei+1}else{si=ni;rt+=t.substring(pi,ni)}pi=t.indexOf(ESP,si)}if(rt.length!=0){rt+=t.substring(si);el.innerHTML=rt}}"#);
        doc.push_str("\n</script>\n</head>\n<body>\n<div class=\"container\">\n");
        // Format timestamp and JVM version for header
        let date_str = if self.heap.timestamp_millis > 0 {
            chrono::DateTime::from_timestamp_millis(self.heap.timestamp_millis as i64)
                .map(|dt| {
                    let local: chrono::DateTime<chrono::Local> = dt.into();
                    local.format("%a %b %d %H:%M:%S %Z %Y").to_string()
                })
                .unwrap_or_else(|| "N/A".to_string())
        } else {
            "N/A".to_string()
        };
        let jvm_info = self.stats.jvm_info.get();
        let jvm_version = jvm_info.system_properties.iter()
            .find(|(k, _)| k == "java.version")
            .map(|(_, v)| v.as_str())
            .unwrap_or("N/A");

        doc.push_str(&format!(
            "<header class=\"report-header\"><h1>SS-Mat {} Heap Dump Report</h1><div class=\"meta\">{}<br>Created on {} &middot; JVM {}</div></header>\n",
            "0.1.0 (Rust)", self.dump_file_name, date_str, jvm_version
        ));
        doc.push_str(&body);
        doc.push_str("\n<footer class=\"report-footer\">SS-Mat Heap Dump Analyzer (Rust)</footer>\n</div>\n</body>\n</html>");

        Ok(doc)
    }

    // ---- Section implementations ----

    fn begin_section(&mut self, html: &mut HtmlStringBuilder, title: &str, open: bool, indicator: &str) {
        if self.section_count > 0 {
            html.append_details_close_tag();
        }
        let id = self.next_section_id();
        let open_attr = if open { "open" } else { "" };
        let indicator_str = if indicator.is_empty() {
            String::new()
        } else {
            format!(" {}", indicator)
        };
        let header = format_section_header(
            &format!("{}{}{}", self.section_numbered_title(), title, indicator_str),
            None,
        );
        html.append_raw(&format!(
            "<details {} id=\"{}\">\n<summary>{}</summary><br>\n",
            open_attr, id, header
        ));
    }

    fn section_top_level_stats(&mut self, html: &mut HtmlStringBuilder) {
        self.begin_section(html, "Top-Level Stats", true, "");

        // Count instance types
        let mut obj_array_count: u64 = 0;
        let mut prim_array_count: u64 = 0;
        let mut instance_count: u64 = 0;
        let mut obj_array_bytes: i64 = 0;
        let mut prim_array_bytes: i64 = 0;
        let mut instance_bytes: i64 = 0;

        for pc in &self.stats.per_class {
            let name = &pc.class_name;
            let inst = pc.instance_count.load(Relaxed);
            let inst_bytes = pc.total_instance_bytes.load(Relaxed);
            let arr_count = pc.array_live_count.load(Relaxed) as u64;
            let arr_bytes = pc.array_live_bytes.load(Relaxed);

            if name.starts_with('[') {
                if name.contains('L') || name.starts_with("[[") {
                    obj_array_count += arr_count;
                    obj_array_bytes += arr_bytes;
                } else {
                    prim_array_count += arr_count;
                    prim_array_bytes += arr_bytes;
                }
            } else if name.ends_with("[]") {
                let inner = &name[..name.len() - 2];
                if matches!(inner, "byte" | "boolean" | "char" | "short" | "int" | "long" | "float" | "double") {
                    prim_array_count += arr_count;
                    prim_array_bytes += arr_bytes;
                } else {
                    obj_array_count += arr_count;
                    obj_array_bytes += arr_bytes;
                }
            } else {
                instance_count += inst;
                instance_bytes += inst_bytes;
            }
        }

        let total_count = instance_count + obj_array_count + prim_array_count;
        let total_bytes = instance_bytes + obj_array_bytes + prim_array_bytes;
        let class_count = self.heap.classes.len();
        let ptr_size = self.heap.oop_size;
        let header_size = self.heap.header_size;
        let gc_roots = self.heap.roots.len();

        // Table 1: Objects by type
        html.begin_table(true, 20, &["", "Instances", "Object arrays", "Primitive arrays", "Total"]);
        html.append_table_row(&[
            "<b>Objects</b>",
            &format_count(instance_count),
            &format_count(obj_array_count),
            &format_count(prim_array_count),
            &format_count(total_count),
        ]);
        html.append_table_row(&[
            "<b>Bytes</b>",
            &format_with_percent(instance_bytes, total_bytes.max(1)),
            &format_with_percent(obj_array_bytes, total_bytes.max(1)),
            &format_with_percent(prim_array_bytes, total_bytes.max(1)),
            &format_with_percent(total_bytes, total_bytes),
        ]);
        html.end_table();

        // Table 2: Live vs Garbage
        let garbage_bytes = self.stats.total_garbage_bytes.load(Relaxed);
        let garbage_count = self.stats.total_garbage_count.load(Relaxed);
        let live_count = total_count - garbage_count;
        let live_bytes = total_bytes - garbage_bytes;

        html.append_raw("<div>\n");
        html.begin_table(true, 20, &["", "Live", "Garbage", "Total objects"]);
        html.append_table_row(&[
            "<b>Objects</b>",
            &format_count(live_count),
            &format_count(garbage_count),
            &format_count(total_count),
        ]);
        html.append_table_row(&[
            "<b>Bytes</b>",
            &format_with_percent(live_bytes, total_bytes.max(1)),
            &format_with_percent(garbage_bytes, total_bytes.max(1)),
            &format_with_percent(total_bytes, total_bytes),
        ]);
        html.end_table();
        html.append_raw("</div>\n");

        // Table 3: Class/Thread/GC root counts
        // Thread count = number of stack trace records in HPROF (all threads at dump time),
        // matching Java's heapDump.stringTable.size()
        let thread_count = self.heap.stack_traces.len() as u64;
        html.append_raw("<div>\n");
        html.begin_table(true, 20, &["Number of classes", "Number of threads", "Number of GC roots"]);
        html.append_table_row(&[
            &format_count(class_count as u64),
            &format_count(thread_count),
            &format_count(gc_roots as u64),
        ]);
        html.end_table();
        html.append_raw("</div>\n");

        // Table 4: JVM internals
        html.append_raw("<div>\n");
        html.begin_table(true, 20, &["JVM pointer size", "Object header size", "Object alignment"]);
        html.append_table_row(&[
            &format!("{} bytes", ptr_size),
            &format!("{} bytes", header_size),
            "8 bytes",
        ]);
        html.end_table();
        html.append_raw("</div>\n");
    }

    fn section_oom_thread(&mut self, html: &mut HtmlStringBuilder) {
        // Find the OOM thread: native method at top, but skip idle threads (wait/park/sleep)
        let thread_info: Vec<_> = self.stats.thread_info.lock().unwrap().clone();

        let idle_methods: &[&str] = &[
            "wait", "park", "parkNanos", "parkUntil",
            "sleep", "yield", "waitForReferencePendingList",
        ];

        let mut oom_thread_serial: Option<u32> = None;
        let mut oom_frame_count: usize = 0;

        // Iterate in serial order so ties on stack depth resolve deterministically
        // (and match the order Java walks the thread list in).
        let mut serials: Vec<u32> = self.heap.stack_traces.keys().copied().collect();
        serials.sort_unstable();
        for serial in serials {
            let st = &self.heap.stack_traces[&serial];
            if let Some(first_frame) = st.frames.first() {
                if first_frame.line_number == -3 {
                    let is_idle = idle_methods.iter().any(|m| first_frame.method_name == *m);
                    if !is_idle && st.frames.len() > oom_frame_count {
                        oom_thread_serial = Some(serial);
                        oom_frame_count = st.frames.len();
                    }
                }
            }
        }

        if let Some(serial) = oom_thread_serial {
            let stack = self.heap.stack_traces.get(&serial).unwrap().clone();
            let ti = thread_info.iter().find(|t| t.thread_serial == serial);
            let thread_name = ti.map(|t| t.name.as_str()).unwrap_or("<unknown>");
            let is_daemon = ti.map(|t| t.is_daemon).unwrap_or(false);

            // Get pre-extracted local variable details
            let oom_locals: Vec<_> = self.stats.oom_frame_locals.lock().unwrap().clone();

            // Build a lookup: frame_depth -> &[LocalVarDetail]
            let mut locals_by_depth: std::collections::HashMap<u32, &Vec<LocalVarDetail>> = std::collections::HashMap::new();
            for fl in &oom_locals {
                locals_by_depth.insert(fl.frame_depth, &fl.objects);
            }

            self.begin_section(html, "Thread throwing OutOfMemoryError", true, "<font class=\"OvhdColorHigh\">&nbsp;found&nbsp;</font>");
            html.append_raw("<b>This dump was created after OutOfMemoryError in the following thread:</b><br>");
            html.append_raw(&format!(
                "<br><b>Thread name: \"{}\", daemon: {}</b><br>",
                thread_name, is_daemon
            ));

            html.append_raw("<br>");
            for (depth, frame) in stack.frames.iter().enumerate() {
                let line_info = match frame.line_number {
                    -1 => format!("unknown, ID={}, line not available", frame.frame_id),
                    -2 => format!("compiled method, ID={}", frame.frame_id),
                    -3 => "Native method".to_string(),
                    n => format!("{}:{}, ID={}", frame.source_file, n, frame.frame_id),
                };
                let display_class = if frame.class_name.is_empty() || frame.class_name == "unknown" {
                    "null".to_string()
                } else {
                    frame.class_name.replace('/', ".")
                };
                html.append_raw(&format!(
                    "<t-c>{}</t-c>.<t-m>{}</t-m>({})<br>",
                    display_class,
                    frame.method_name,
                    line_info,
                ));

                // Show local variables for this frame depth in collapsible details
                if let Some(objects) = locals_by_depth.get(&(depth as u32)) {
                    if !objects.is_empty() {
                        html.append_raw("<details>\n<summary>&nbsp<b> Local variables</b></summary>\n");
                        for obj in *objects {
                            html.append_raw(&format_local_var_detail(obj));
                            html.append_raw("<br>\n");
                        }
                        html.append_raw("</details>");
                    }
                }
            }
        } else {
            self.begin_section(html, "Thread throwing OutOfMemoryError", false, "");
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&[
                "No OutOfMemoryError thread detected in this heap dump.",
                "",
            ]);
            html.end_table();
        }
    }

    fn section_memory_by_class(&mut self, html: &mut HtmlStringBuilder) {
        self.begin_section(html, "Where Memory Goes, by Class", false, "");

        // Section description (matching Java JXRay)
        html.append_raw("Instances of some class(es) take up a big percentage of memory.<br>\n");
        html.append_raw("That can happen when the number of instances is very high, or their size<br>\n");
        html.append_raw("is very large (e.g. for arrays). Depending on whether all these instances<br>\n");
        html.append_raw("are really necessary or not, and also whether they are live or garbage,<br>\n");
        html.append_raw("this may or may not be a problem. See other report sections for more details<br>\n");
        html.append_raw("about these objects.<br>\n");

        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);

        // Include both instances and arrays. `total_instance_bytes` covers
        // instances only; arrays are accumulated separately by the linear scan.
        let mut classes: Vec<(i64, i64, &ss_stats::per_class::PerClassStats)> = self
            .stats
            .per_class
            .iter()
            .map(|pc| {
                let inst_bytes = pc.total_instance_bytes.load(Relaxed);
                let arr_bytes = pc.array_live_bytes.load(Relaxed);
                let shallow = inst_bytes + arr_bytes;
                let live = (inst_bytes - pc.instance_garbage_bytes.load(Relaxed)).max(0)
                    + (arr_bytes - pc.array_garbage_bytes.load(Relaxed)).max(0);
                (shallow, live, pc)
            })
            .filter(|(shallow, _, _)| *shallow > 0)
            .collect();
        classes.sort_by(|a, b| b.0.cmp(&a.0));
        let display_count = classes.len().min(500);

        let headers: &[(&str, &str)] = &[
            (
                "# instances",
                "How many instances of the given class there are in this dump.",
            ),
            (
                "Shallow size",
                "How much memory is used by all instances of the given class.",
            ),
            (
                "Impl-incl. size",
                "Implementation-inclusive size. This is bigger than shallow size for collections, Strings and a few other classes. It includes the size of implementation, for example memory used by internal HashMap$Node objects in HashMap, internal char[] array in String, etc.",
            ),
            (
                "Live shallow",
                "Shallow size of reachable (live) instances and arrays of this class. Full dominator-based retained size is not implemented yet.",
            ),
            ("Class name", ""),
        ];
        html.begin_table_with_tooltips(true, display_count, headers);

        let ref_chains = self.stats.ref_chains.lock().unwrap();

        for (class_rank, &(shallow, live_shallow, pc)) in classes.iter().take(500).enumerate() {
            let inst = pc.instance_count.load(Relaxed) + pc.array_live_count.load(Relaxed) as u64;
            // Impl-incl. size: shallow + duplicate string bytes overhead for this class
            let impl_incl = shallow + pc.duplicate_string_bytes.load(Relaxed).max(0);
            let class_name = pc.class_name.replace('/', ".");

            html.append_table_row(&[
                &format_count(inst),
                &format_with_percent(shallow, total_bytes),
                &format_with_percent(impl_incl, total_bytes),
                &format_with_percent(live_shallow, total_bytes),
                &format!("<t-c>{}</t-c>", class_name),
            ]);

            // Render reference chains if available for this class
            if let Some(chains) = ref_chains.chains.get(&pc.class_index) {
                if !chains.expensive_fields.is_empty() || !chains.full_chains.is_empty() {
                    let chain_html = render_class_ref_chains(chains, total_bytes);
                    if !chain_html.is_empty() {
                        // Collect sample object ids so we can also render
                        // per-instance chains with field values. To keep the
                        // report size bounded (matching Java, which only
                        // renders per-object chains for a few high-memory
                        // classes), limit to the top 15 classes and 3 samples each.
                        let mut sample_ids: Vec<u64> = Vec::new();
                        if class_rank < 15 {
                            for g in chains.full_chains.iter().chain(chains.expensive_fields.iter()) {
                                for &id in g.sample_obj_ids.iter().take(3) {
                                    if id != 0 {
                                        sample_ids.push(id);
                                    }
                                    if sample_ids.len() >= 3 {
                                        break;
                                    }
                                }
                                if sample_ids.len() >= 3 {
                                    break;
                                }
                            }
                        }
                        // Append per-instance chains (with decoded field values)
                        // before the class-level chain panel.
                        let mut combined = String::new();
                        if !sample_ids.is_empty() {
                            self.render_instance_ref_chains_into(
                                &mut combined,
                                "Per-instance reference chains",
                                shallow,
                                total_bytes,
                                &sample_ids,
                                None,
                            );
                        }
                        combined.push_str(&format!(
                            "<details><summary>Reference chains</summary>{}</details>",
                            chain_html,
                        ));
                        html.append_span_row(&combined, 5, false);
                    }
                }
            }
        }
        html.end_table();
    }

    fn section_memory_by_gc_root(&mut self, html: &mut HtmlStringBuilder) {
        self.begin_section(html, "Where Memory Goes, by GC Root", false, "");

        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        let gc_roots = self.stats.gc_roots.lock().unwrap();

        // Main table: per-root entries with collapsible reference trees
        html.begin_table(true, gc_roots.entries.len() + 2, &[]);
        html.append_raw("<tr class=\"C_01\"><th class=\"TAL\">&nbsp;Retained Size</th><th class=\"TAL\">Root Tree</th></tr>\n");

        // Garbage (unreachable objects) entry first
        if gc_roots.garbage.count > 0 {
            let gb = &gc_roots.garbage;
            let pct = calc_percent(gb.bytes, total_bytes);
            html.append_raw(&format!(
                "<tr class=\"C_02\"><td class=\"TAL\">&nbsp;{} ({})</td><td class=\"TAL\">",
                format_bytes(gb.bytes),
                format_percent(pct),
            ));
            html.append_raw(&format!(
                "<details><summary>{} ({}) Unreachable (garbage) objects</summary>",
                format_bytes(gb.bytes),
                format_percent(pct),
            ));
            // Garbage top classes as a simple list
            html.append_raw("<ol>");
            for (name, count, bytes) in &gb.top_classes {
                let cpct = calc_percent(*bytes, total_bytes);
                html.append_raw(&format!(
                    "<li><t-c>{}</t-c>&nbsp;{}&nbsp;({}),&nbsp;{}&nbsp;object(s)</li>\n",
                    name, format_bytes(*bytes), format_percent(cpct), format_count(*count),
                ));
            }
            if gb.top_classes.len() < gb.count as usize {
                let shown = gb.top_classes.len() as u64;
                html.append_raw(&format!(
                    "<li>... and {} more garbage object types</li>\n",
                    format_count(gb.count - shown),
                ));
            }
            html.append_raw("</ol></details></td></tr>\n");
        }

        // Per-root entries
        for entry in &gc_roots.entries {
            let pct = calc_percent(entry.retained_bytes, total_bytes);
            let desc = format_root_description(&entry.meta);

            html.append_raw(&format!(
                "<tr class=\"C_02\"><td class=\"TAL\">&nbsp;{}</td><td class=\"TAL\">",
                format_with_percent(entry.retained_bytes, total_bytes),
            ));

            html.append_raw(&format!(
                "<details><summary>{} ({}) Object tree for GC root(s) {}</summary>",
                format_bytes(entry.retained_bytes),
                format_percent(pct),
                desc,
            ));

            // Root object description
            html.append_raw(&format!(
                "<br><details><summary>Root object <t-c>{}</t-c>@{:x}</summary>",
                entry.meta.class_name, entry.meta.obj_id,
            ));
            html.append_raw(&format!(
                "<t-c>{}</t-c>()</details>",
                entry.meta.class_name,
            ));

            // Reference tree
            html.append_raw("<ol>");
            html.append_raw(&render_tree_node(&entry.tree, total_bytes));
            html.append_raw("</ol>");

            html.append_raw("</details></td></tr>\n");
        }

        // Remaining roots summary
        if gc_roots.remaining_count > 0 {
            let rpct = calc_percent(gc_roots.remaining_bytes, total_bytes);
            html.append_raw(&format!(
                "<tr class=\"C_03\"><td class=\"TAL\">&nbsp;</td><td class=\"TAL\">... and {} more GC roots together retaining {} ({})</td></tr>\n",
                gc_roots.remaining_count,
                format_bytes(gc_roots.remaining_bytes),
                format_percent(rpct),
            ));
        }

        html.end_table();

        // Random sample of remaining roots
        if !gc_roots.remaining_samples.is_empty() {
            html.append_raw(&format!(
                "<br>Random sample of the remaining {} GC roots<br>\n",
                gc_roots.remaining_count,
            ));
            html.append_raw("<div>");
            html.begin_table(true, gc_roots.remaining_samples.len(), &["Root Type", "Class", "Address"]);
            for sample in &gc_roots.remaining_samples {
                let desc = format_root_description(sample);
                html.append_table_row(&[
                    &desc,
                    &format!("<t-c>{}</t-c>", sample.class_name),
                    &format!("{:x}", sample.obj_id),
                ]);
            }
            html.end_table();
            html.append_raw("</div>");
        }
    }

    fn section_live_vs_garbage(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed);
        let garbage_bytes = self.stats.total_garbage_bytes.load(Relaxed);
        let garbage_pct = calc_percent(garbage_bytes, total_bytes);

        let indicator = if garbage_pct > 30.0 {
            severity_med(&format!("Garbage size {:.1}% ({})", garbage_pct, format_bytes(garbage_bytes)))
        } else if garbage_bytes > 0 {
            severity_low(&format!("Garbage size {:.1}% ({})", garbage_pct, format_bytes(garbage_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Live vs Garbage Objects", false, &indicator);

        let total_objects = self.stats.total_instances.load(Relaxed);
        let garbage_count = self.stats.total_garbage_count.load(Relaxed);
        let live_count = total_objects - garbage_count;
        let live_bytes = total_bytes - garbage_bytes;

        // Summary table
        html.begin_table(true, 10, &["", "Count", "Size"]);
        html.append_table_row(&[
            "Live objects",
            &format_count(live_count),
            &format_with_percent(live_bytes, total_bytes.max(1)),
        ]);
        html.append_table_row(&[
            "Garbage (unreachable) objects",
            &format_count(garbage_count),
            &format_with_percent(garbage_bytes, total_bytes.max(1)),
        ]);
        html.append_table_row(&[
            "<b>Total</b>",
            &format_count(total_objects),
            &format_bytes(total_bytes),
        ]);
        html.end_table();

        // Per-class breakdown
        let mut class_data: Vec<(&PerClassStats, u64, i64, u64, i64)> = Vec::new();
        for pc in &self.stats.per_class {
            let inst = pc.instance_count.load(Relaxed);
            let inst_bytes = pc.total_instance_bytes.load(Relaxed);
            let garb_count = pc.instance_garbage_count.load(Relaxed) as u64;
            let garb_bytes = pc.instance_garbage_bytes.load(Relaxed);
            // Also count array garbage
            let arr_garb_count = pc.array_garbage_count.load(Relaxed) as u64;
            let arr_garb_bytes = pc.array_garbage_bytes.load(Relaxed);
            let total_garb_count = garb_count + arr_garb_count;
            let total_garb_bytes = garb_bytes + arr_garb_bytes;
            let live_inst = inst.saturating_sub(garb_count);
            let live_arr = pc.array_live_count.load(Relaxed) as u64;
            let live_arr_count = live_arr.saturating_sub(arr_garb_count);
            let total_live = live_inst + live_arr_count;
            let live_arr_bytes = (pc.array_live_bytes.load(Relaxed) as i64).saturating_sub(arr_garb_bytes);
            let total_live_bytes = inst_bytes.saturating_sub(garb_bytes) + live_arr_bytes;

            if total_garb_count > 0 || (total_live > 0 && pc.class_name.starts_with('[') || pc.class_name.ends_with("[]")) {
                class_data.push((pc, total_garb_count, total_garb_bytes, total_live, total_live_bytes));
            }
        }

        if !class_data.is_empty() {
            class_data.sort_by(|a, b| b.2.cmp(&a.2));
            html.append_raw("<br>Details:<br>\n<div class=\"scroll-wrap\">\n");
            // Header row with tooltips (matching Java JXRay style).
            html.append_raw("<table class=\"dt\" id=\"Table-3\">\n");
            html.append_raw("<tr class=\"C_01\">");
            html.append_raw("<th onclick=\"sortTable(3, 0)\">&nbsp<b>&nbsp<span title=\"Number of garbage (unreachable) instances of this class in the dump.\"># instances garbage&nbsp;&#8593;&#8595;&nbsp;<i-s>&#x24d8;</i-s></span>&nbsp</b></th>");
            html.append_raw("<th onclick=\"sortTable(3, 1)\">&nbsp<b>&nbsp<span title=\"Shallow memory size of all garbage instances of this class.\">Shallow size garbage&nbsp;&#8593;&#8595;&nbsp;<i-s>&#x24d8;</i-s></span>&nbsp</b></th>");
            html.append_raw("<th onclick=\"sortTable(3, 2)\">&nbsp<b>&nbsp<span title=\"Number of live (reachable) instances of this class.\"># instances live&nbsp;&#8593;&#8595;&nbsp;<i-s>&#x24d8;</i-s></span>&nbsp</b></th>");
            html.append_raw("<th onclick=\"sortTable(3, 3)\">&nbsp<b>&nbsp<span title=\"Shallow memory size of all live instances of this class.\">Shallow size live&nbsp;&#8593;&#8595;&nbsp;<i-s>&#x24d8;</i-s></span>&nbsp</b></th>");
            html.append_raw("<th class=\"TAL\"><b>&nbsp;Class name&nbsp;</b></th>");
            html.append_raw("</tr>\n");

            let mut shown = 0;
            for (pc, garb_cnt, garb_bytes, live_cnt, live_bytes) in &class_data {
                if shown >= 50 { break; }
                if *garb_cnt == 0 && *garb_bytes == 0 { continue; }
                let row_class = if shown % 2 == 0 { "C_02" } else { "C_03" };
                html.append_raw(&format!(
                    "<tr class=\"{}\"><td>&nbsp;{}</td><td>&nbsp;{}</td><td>&nbsp;{}</td><td>&nbsp;{}</td><td class=\"TAL\">&nbsp;{}</td></tr>\n",
                    row_class,
                    format_count(*garb_cnt as u64),
                    format_with_percent(*garb_bytes, total_bytes.max(1)),
                    format_count(*live_cnt as u64),
                    format_with_percent(*live_bytes, total_bytes.max(1)),
                    format_class_name(&pc.class_name),
                ));
                shown += 1;
            }
            html.append_raw("</table>\n");
            html.append_raw("</div>\n");
        }
    }

    fn section_fixed_overhead(&mut self, html: &mut HtmlStringBuilder) {
        let header_size = self.heap.header_size as i64;
        let total_objects = self.stats.total_instances.load(Relaxed);
        let total_bytes = self.stats.total_shallow_size.load(Relaxed);
        let overhead_bytes = total_objects as i64 * header_size;
        let overhead_pct = calc_percent(overhead_bytes, total_bytes);

        let indicator = if overhead_pct > 10.0 {
            severity_med(&format!("Overhead {:.1}% ({})", overhead_pct, format_bytes(overhead_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Fixed per-object overhead", false, &indicator);

        html.begin_table(true, 10, &["", ""]);
        html.append_table_row(&[
            "Object header size",
            &format!("{} bytes", header_size),
        ]);
        html.append_table_row(&[
            "Total # objects",
            &format_count(total_objects),
        ]);
        html.append_table_row(&[
            "Total header overhead",
            &format_with_percent(overhead_bytes, total_bytes.max(1)),
        ]);
        html.end_table();

        // Per-class overhead breakdown
        let mut class_overhead: Vec<(&PerClassStats, i64)> = Vec::new();
        for pc in &self.stats.per_class {
            let inst = pc.instance_count.load(Relaxed);
            if inst > 0 {
                let oh = inst as i64 * header_size;
                if oh > 1024 {
                    class_overhead.push((pc, oh));
                }
            }
            let arr_count = pc.array_live_count.load(Relaxed) as u64;
            if arr_count > 0 {
                let oh = arr_count as i64 * header_size;
                if oh > 1024 {
                    // Add as separate entry for array overhead
                    class_overhead.push((pc, oh));
                }
            }
        }
        class_overhead.sort_by(|a, b| b.1.cmp(&a.1));

        if !class_overhead.is_empty() {
            html.append_raw("<details><summary>Per-class overhead breakdown</summary>\n<div class=\"scroll-wrap\">\n");
            html.begin_table(true, 0, &["#instances", "Overhead", "Shallow size", "Overhead %", "Class name"]);
            for (pc, oh) in class_overhead.iter().take(50) {
                let inst = pc.instance_count.load(Relaxed);
                let inst_bytes = pc.total_instance_bytes.load(Relaxed);
                let arr_count = pc.array_live_count.load(Relaxed) as u64;
                let arr_bytes = pc.array_live_bytes.load(Relaxed);
                let count = inst + arr_count;
                let shallow = inst_bytes + arr_bytes;
                let pct = calc_percent(*oh, total_bytes.max(1));
                html.append_table_row(&[
                    &format_count(count),
                    &format_bytes(*oh),
                    &format_bytes(shallow),
                    &format!("{:.1}%", pct),
                    &format_class_name(&pc.class_name),
                ]);
            }
            html.end_table();
            html.append_raw("</div>\n</details>\n");
        }
    }

    fn section_finalization(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        self.begin_section(html, "Objects Awaiting Finalization", false, "");

        // Look for java/lang/ref/Finalizer class
        let finalizer_count = self.heap.class_by_name.get("java/lang/ref/Finalizer")
            .and_then(|&idx| self.stats.get_per_class(idx))
            .map(|pc| pc.instance_count.load(Relaxed))
            .unwrap_or(0);

        if finalizer_count > 0 {
            html.begin_table(true, 5, &["Finalizer queue objects", "Count"]);
            html.append_table_row(&[
                "java.lang.ref.Finalizer instances",
                &format_count(finalizer_count),
            ]);
            html.end_table();
            if finalizer_count > 1000 {
                html.append_raw(&format!("<br>{}", severity_med(&format!(
                    "Large finalizer queue ({}) may indicate finalizer-related memory issues.",
                    format_count(finalizer_count)
                ))));
            }
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&[
                "No objects awaiting finalization detected.",
                "",
            ]);
            html.end_table();
        }

        // Typical reference paths for Finalizer/Finalizer$FinalizerNode classes.
        let _ = html.append_raw(&self.render_class_ref_chain_panel("java.lang.ref.Finalizer", total_bytes));

        // Per-instance ref chain for sample Finalizer instances.
        if finalizer_count > 0 {
            let sample_ids: Vec<u64> = {
                let guard = self.stats.traversal_data.lock().unwrap();
                if let Some(data) = guard.as_ref() {
                    if let Some(&class_idx) = self.heap.class_by_name.get("java/lang/ref/Finalizer") {
                        data.sample_ids_by_class(class_idx, 20)
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            self.render_instance_ref_chains(
                html,
                "Reference chains holding Finalizer queue instances",
                finalizer_count as i64 * 64,
                total_bytes,
                &sample_ids,
                None,
            );
        }
    }

    fn section_duplicate_strings(&mut self, html: &mut HtmlStringBuilder) {
        let report = self.stats.string_dup.get_report();
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        let pct = calc_percent(report.wasted_bytes, total_bytes);

        let indicator = if report.wasted_bytes > 0 && pct > 1.0 {
            severity_med(&format!("Overhead {:.1}% ({})", pct, format_bytes(report.wasted_bytes)))
        } else if report.wasted_bytes > 0 {
            severity_low(&format!("Overhead {:.1}% ({})", pct, format_bytes(report.wasted_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Duplicate Strings", false, &indicator);

        // Summary table
        html.begin_table(true, 5, &["Total strings", "Unique strings", "Duplicate values", "Overhead"]);
        html.append_table_row(&[
            &format_count(report.total_strings as u64),
            &format_count(report.unique_strings as u64),
            &format_count(report.duplicate_group_count as u64),
            &format_with_percent(report.wasted_bytes, total_bytes),
        ]);
        html.end_table();

        // Top duplicate groups detail table
        if !report.top_groups.is_empty() {
            html.append_raw("<details><summary>Top duplicate string values</summary><div class=\"scroll-wrap\">");
            html.begin_table(true, 0, &["Overhead", "# objects", "Value"]);
            for g in report.top_groups.iter().take(100) {
                let val_preview = if g.content.len() > 200 {
                    format!("{}...", &g.content[..200])
                } else {
                    g.content.clone()
                };
                let escaped = val_preview
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
                    .replace('"', "&quot;");
                html.append_table_row(&[
                    &format_with_percent(g.wasted_bytes, total_bytes),
                    &format_count(g.count as u64),
                    &escaped,
                ]);
            }
            html.end_table();
            html.append_raw("</div></details>");
        }

        // "What can I do to fix?" panel
        if report.wasted_bytes > 0 {
            html.append_raw("<details><summary><b>What can I do to fix?</b></summary>\n<div>\n");
            html.append_raw("<table class=\"dt\"><tr><td class=\"TAL\"><font class=\"Bigger\">\
                The overhead (waste) of all duplicate strings is <b>");
            html.append_raw(&format_with_percent(report.wasted_bytes, total_bytes));
            html.append_raw("</b>. That's how much memory you could save if there were no duplicate strings.<br><br>\
                <b>How to fix:</b><br>\
                - Use <code>String.intern()</code> for frequently duplicated strings (but be aware of PermGen/Metaspace implications)<br>\
                - Enable <code>-XX:+UseStringDeduplication</code> with G1 GC (Java 8u20+)<br>\
                - Consider using static final constants for known duplicate values<br>\
                - For SQL statements, use PreparedStatement pooling instead of string concatenation\
                </font></td></tr></table>\n");
            html.append_raw("</div>\n</details>\n");
        }

        // Typical reference paths for java.lang.String (uses per-class ref_chains data).
        let _ = html.append_raw(&self.render_class_ref_chain_panel("java.lang.String", total_bytes));

        // Per-instance reference chains for sample duplicate string objects.
        let mut sample_ids: Vec<u64> = Vec::new();
        for g in report.top_groups.iter().take(10) {
            for &id in g.sample_obj_ids.iter().take(2) {
                sample_ids.push(id);
            }
        }
        self.render_instance_ref_chains(
            html,
            "Per-instance reference chains for sample duplicate Strings",
            report.wasted_bytes,
            total_bytes,
            &sample_ids,
            Some("Reference Chains for Duplicate Strings"),
        );
    }

    fn section_bad_collections(&mut self, html: &mut HtmlStringBuilder) -> std::io::Result<()> {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);

        // Java's Bad Collections overhead = wasted capacity of BAD collections
        // (EMPTY/SINGLE_ELEM/SMALL/SPARSE/MANY_NULLS), reached during the live
        // traversal. We captured EMPTY/SINGLE_ELEM collections during the scan
        // (each contributing its full shallow body as waste). Filter to live.
        // Java's processContainer runs for both live and garbage objects
        // (the second traversal pass), so do not filter to live here.
        // Aggregate bad-collection waste per class. The record vector carries one
        // entry per bad collection — millions of them on a large dump — so it is
        // read under the lock and aggregated into a per-class table instead of
        // being cloned into a second copy of the same size.
        let mut per_class: std::collections::HashMap<u32, (u64, i64)> = std::collections::HashMap::new();
        let mut total_collection_bytes: i64 = 0;
        let mut total_bad_count: u64 = 0;
        {
            let records = self.stats.bad_collection_records.lock().unwrap();
            let store=self.stats.bad_collection_store.lock().unwrap();
            for r in store.as_ref().into_iter().flat_map(|s|s.iter()).chain(records.iter().filter(|_|store.is_none()).cloned().map(Ok)) {
                let r=r?;
                if r.kind == ss_stats::problem_types::COLLECTION_KIND_NONE {
                    continue;
                }
                let (c, b) = per_class.entry(r.class_idx).or_insert((0, 0));
                *c += 1;
                *b += r.waste;
                total_collection_bytes += r.waste;
                total_bad_count += 1;
            }
        }
        let total_collections: u64 = self.stats.per_class.iter()
            .filter(|pc| is_collection_class_name_for_report(&pc.class_name))
            .map(|pc| pc.instance_count.load(Relaxed))
            .sum();
        let mut entries: Vec<(String, u64, i64)> = per_class.iter()
            .map(|(&idx, &(c, b))| {
                let name = self.heap.classes.get(idx as usize)
                    .map(|c| c.name.replace('/', "."))
                    .unwrap_or_else(|| "unknown".to_string());
                (name, c, b)
            })
            .collect();
        entries.sort_by(|a, b| b.2.cmp(&a.2));

        let pct = calc_percent(total_collection_bytes, total_bytes);
        let indicator = if total_collection_bytes > 0 && pct >= 2.0 {
            severity_med(&format!("Overhead {:.1}% ({})", pct, format_bytes(total_collection_bytes)))
        } else if total_collection_bytes > 0 {
            severity_low(&format!("Overhead {:.1}% ({})", pct, format_bytes(total_collection_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Bad Collections", false, &indicator);

        if total_bad_count > 0 {
            html.begin_table(false, 0, &["Total collections", "Bad collections", "Overhead:"]);
            html.append_table_row(&[
                &format_count(total_collections),
                &format_count(total_bad_count),
                &format_with_percent(total_collection_bytes, total_bytes),
            ]);
            html.end_table();

            // Per-class table
            html.begin_table(true, 0, &["# instances", "Wasted bytes", "% of heap", "Class name"]);
            for (name, count, bytes) in entries.iter().take(30) {
                html.append_table_row(&[
                    &format_count(*count),
                    &format_bytes(*bytes),
                    &format!("{:.1}%", calc_percent(*bytes, total_bytes)),
                    name,
                ]);
            }
            html.end_table();

            // Estimate empty collections (heuristic: empty/singleton collections
            // like EmptyList, SingletonList have very small fixed sizes)
            let mut empty_estimates: Vec<(&str, u64, i64)> = Vec::new();
            let empty_classes = [
                ("java/util/Collections$EmptyList", "Empty List (Collections.emptyList())"),
                ("java/util/Collections$EmptyMap", "Empty Map (Collections.emptyMap())"),
                ("java/util/Collections$EmptySet", "Empty Set (Collections.emptySet())"),
                ("java/util/Collections$SingletonList", "Singleton List"),
                ("java/util/Collections$SingletonSet", "Singleton Set"),
            ];
            for (class_name, label) in empty_classes {
                if let Some(&idx) = self.heap.class_by_name.get(class_name) {
                    if let Some(pc) = self.stats.get_per_class(idx) {
                        let count = pc.instance_count.load(Relaxed);
                        let bytes = pc.total_instance_bytes.load(Relaxed);
                        if count > 0 {
                            empty_estimates.push((label, count, bytes));
                        }
                    }
                }
            }

            if !empty_estimates.is_empty() {
                html.append_raw("<br><details><summary>Empty/Singleton collection estimates</summary><div>");
                html.begin_table(true, 0, &["Collection type", "# instances", "Shallow size"]);
                for (label, count, bytes) in &empty_estimates {
                    html.append_table_row(&[
                        label,
                        &format_count(*count),
                        &format_with_percent(*bytes, total_bytes),
                    ]);
                }
                html.end_table();
                html.append_raw("</div></details>");
            }

            // Fix panel
            html.append_raw("<details><summary><b>What can I do to fix?</b></summary>\n<div>\n");
            html.append_raw("<table class=\"dt\"><tr><td class=\"TAL\"><font class=\"Bigger\">\
                Collections can waste memory in several ways:<br>\
                <br><b>1. Empty collections:</b> Many applications create thousands of empty HashMap/ArrayList instances. \
                Consider using <code>Collections.emptyMap()</code>, <code>Collections.emptyList()</code>, \
                or lazy initialization (create the collection only when first needed).<br>\
                <br><b>2. Oversized collections:</b> A HashMap with default initial capacity 16 but only 1-2 entries \
                wastes the backing array space. Use constructors with initial capacity hints: \
                <code>new HashMap<>(expectedSize)</code>. For ArrayList, use <code>new ArrayList<>(size)</code> \
                when the approximate size is known.<br>\
                <br><b>3. HashMap node overhead:</b> Each HashMap entry creates a Node object (32+ bytes). \
                For small maps, consider <code>EnumMap</code>, <code>RegularImmutableMap</code>, or plain arrays.<br>\
                <br><b>Note:</b> Detailed per-collection overhead analysis (empty vs sparsely-filled) \
                requires container walker integration which is planned for a future release.\
                </font></td></tr></table>\n");
            html.append_raw("</div>\n</details>\n");

            // Per-instance ref chain for top collection class.
            if let Some((top_name, _, _)) = entries.first() {
                let lookup = top_name.replace('.', "/");
                let _ = html.append_raw(&self.render_class_ref_chain_panel(top_name, total_bytes));

                // Scan traversal_data for sample instances of the top collection class.
                let sample_ids: Vec<u64> = {
                    let guard = self.stats.traversal_data.lock().unwrap();
                    if let Some(data) = guard.as_ref() {
                        let key: &str = lookup.as_str();
                        if let Some(&class_idx) = self.heap.class_by_name.get(key) {
                            data.sample_ids_by_class(class_idx, 20)
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                };
                self.render_instance_ref_chains(
                    html,
                    &format!("Per-instance reference chains for sample <t-c>{}</t-c> instances", top_name),
                    total_collection_bytes,
                    total_bytes,
                    &sample_ids,
                    Some("Reference Chains for Bad Collections"),
                );
            }
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No collection instances found", ""]);
            html.end_table();
        }
        Ok(())
    }

    fn section_bad_obj_arrays(&mut self, html: &mut HtmlStringBuilder) -> std::io::Result<()> {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);

        // Java only counts bad object arrays reached during the LIVE reachability
        // traversal (processingLiveObjects=true). The linear scan captured ALL
        // bad arrays (live+garbage); filter to live-only here using BFS `visited`.
        // Java's container traversal marks every container backing array visited
        // before processObjectArray(); those arrays never enter Section 10. The
        // per-array records are aggregated under the lock: hundreds of millions of
        // bytes on a large dump, and a clone plus a filtered copy doubled them.
        let mut kind_count = [0u64; 5];
        let mut kind_bytes = [0i64; 5];
        let mut sample_ids: Vec<u64> = Vec::with_capacity(15);
        {
            let records = self.stats.bad_array_records.lock().unwrap();
            let store = self.stats.bad_array_store.lock().unwrap();
            let container_backing = self.stats.container_backing_ids.lock().unwrap();
            for r in store.as_ref().into_iter().flat_map(|s| s.iter()).chain(records.iter().filter(|_| store.is_none()).cloned().map(Ok)) {
                let r = r?;
                if container_backing.contains(&r.obj_id) {
                    continue;
                }
                let kind = r.kind as usize;
                if kind < kind_count.len() {
                    kind_count[kind] += 1;
                    kind_bytes[kind] += r.size;
                }
                if sample_ids.len() < 15 {
                    sample_ids.push(r.obj_id);
                }
            }
        }
        sample_ids.dedup();

        // Java S10's total = sum of instanceCount for classes with
        // problemObjectCount==2, i.e. object arrays excluding multi-dimensional
        // primitive arrays like [[I / [[B.
        let total_object_arrays: u64 = self
            .stats
            .per_class
            .iter()
            .filter(|pc| is_object_array_class_name(&pc.class_name))
            .map(|pc| pc.array_live_count.load(Relaxed) as u64)
            .sum();

        let empty_count = kind_count[0];
        let empty_bytes = kind_bytes[0];
        let single_count = kind_count[1];
        let single_bytes = kind_bytes[1];
        let all_null_count = kind_count[2];
        let all_null_bytes = kind_bytes[2];
        let one_nonnull_count = kind_count[3];
        let one_nonnull_bytes = kind_bytes[3];
        let sparse_count = kind_count[4];
        let sparse_bytes = kind_bytes[4];

        let total_bad = empty_count + single_count + all_null_count + one_nonnull_count + sparse_count;
        let total_bad_bytes = empty_bytes + single_bytes + all_null_bytes + one_nonnull_bytes + sparse_bytes;

        let indicator = if total_bad > 0 && calc_percent(total_bad_bytes, total_bytes) >= 1.0 {
            severity_med(&format!("Overhead {:.1}% ({})", calc_percent(total_bad_bytes, total_bytes), format_bytes(total_bad_bytes)))
        } else if total_bad > 0 {
            severity_low(&format!("{}", format_bytes(total_bad_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Bad Object Arrays", false, &indicator);

        if total_bad > 0 {
            html.begin_table(false, 0, &["Total object arrays", "Bad object arrays", "Overhead"]);
            html.append_table_row(&[
                &format_count(total_object_arrays),
                &format_count(total_bad),
                &format_with_percent(total_bad_bytes, total_bytes),
            ]);
            html.end_table();

            html.begin_table(true, 0, &["Problem type", "# arrays", "Size"]);
            if empty_count > 0 {
                html.append_table_row(&[
                    "Empty object arrays (length=0)",
                    &format_count(empty_count),
                    &format_with_percent(empty_bytes, total_bytes),
                ]);
            }
            if single_count > 0 {
                html.append_table_row(&[
                    "Single-element object arrays (length=1)",
                    &format_count(single_count),
                    &format_with_percent(single_bytes, total_bytes),
                ]);
            }
            if all_null_count > 0 {
                html.append_table_row(&[
                    "All-null object arrays (length>1, all elements null)",
                    &format_count(all_null_count),
                    &format_with_percent(all_null_bytes, total_bytes),
                ]);
            }
            if one_nonnull_count > 0 {
                html.append_table_row(&[
                    "One-element object arrays (length>1, 1 non-null)",
                    &format_count(one_nonnull_count),
                    &format_with_percent(one_nonnull_bytes, total_bytes),
                ]);
            }
            if sparse_count > 0 {
                html.append_table_row(&[
                    "Sparse object arrays (>70% null elements)",
                    &format_count(sparse_count),
                    &format_with_percent(sparse_bytes, total_bytes),
                ]);
            }
            html.end_table();

            html.append_raw("<details><summary><b>What can I do to fix?</b></summary>\n<div>\n");
            html.append_raw("<table class=\"dt\"><tr><td class=\"TAL\"><font class=\"Bigger\">\
                Empty and all-null object arrays waste memory because each array still has an object header. \
                Consider using null references instead of empty arrays, or reuse shared empty array instances \
                (e.g., <code>Collections.emptyList().toArray()</code>).<br>\
                Single-element and one-non-null arrays should ideally be replaced with direct object references when possible.<br>\
                Sparse arrays indicate underutilized backing arrays, common in hash tables with low load factors or \
                pre-allocated collections that were never filled.\
                </font></td></tr></table>\n");
            html.append_raw("</div>\n</details>\n");

            // Per-instance ref chain for sample bad object arrays: the first live
            // bad arrays, collected with the aggregates above.
            self.render_instance_ref_chains(
                html,
                "Per-instance reference chains for sample bad object arrays",
                total_bad_bytes,
                total_bytes,
                &sample_ids,
                Some("Reference Chains for Bad Object Arrays"),
            );
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No bad object arrays found.", ""]);
            html.end_table();
        }
        Ok(())
    }

    fn section_bad_prim_arrays(&mut self, html: &mut HtmlStringBuilder) -> std::io::Result<()> {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        // Aggregates over the per-array records, taken under the lock: there is one
        // record per bad primitive array, hundreds of millions of bytes on a large
        // dump, and a clone would keep a second copy alive for the sake of a few sums.
        let (total_bad, total_bad_bytes, categories, groups, sample_ids) = {
            let records = self.stats.bad_prim_array_records.lock().unwrap();
            let store = self.stats.bad_prim_array_store.lock().unwrap();
            let mut kind_tally = [(0u64, 0i64); 6];
            let mut by_class: std::collections::HashMap<(u32, u8), (u64, i64)> =
                std::collections::HashMap::new();
            let mut total_bad_bytes: i64 = 0;
            for r in store.as_ref().into_iter().flat_map(|s| s.iter()).chain(records.iter().filter(|_| store.is_none()).cloned().map(Ok)) {
                let r = r?;
                total_bad_bytes += r.waste;
                let kind = r.kind as usize;
                if kind < kind_tally.len() {
                    kind_tally[kind].0 += 1;
                    kind_tally[kind].1 += r.waste;
                }
                let entry = by_class.entry((r.class_idx, r.kind)).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += r.waste;
            }
            // Java's processPrimitiveArray categories, sorted by waste.
            let mut categories: Vec<(u8, u64, i64)> = kind_tally
                .iter()
                .enumerate()
                .map(|(kind, &(count, bytes))| (kind as u8, count, bytes))
                .filter(|(_, count, _)| *count > 0)
                .collect();
            categories.sort_by(|a, b| b.2.cmp(&a.2));

            // Per-class + per-problem breakdown.
            let mut groups: Vec<((u32, u8), (u64, i64))> = by_class.into_iter().collect();
            groups.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));

            // The problem arrays the report renders chains for: the ten largest by
            // waste, keeping record order among equal sizes, which is what the
            // full sort used to give.
            let mut best: Vec<(i64, u64)> = Vec::with_capacity(11);
            for r in store.as_ref().into_iter().flat_map(|s| s.iter()).chain(records.iter().filter(|_| store.is_none()).cloned().map(Ok)) {
                let r = r?;
                if best.len() == 10 && r.waste <= best[9].0 {
                    continue;
                }
                let pos = best
                    .iter()
                    .position(|&(waste, _)| waste < r.waste)
                    .unwrap_or(best.len());
                best.insert(pos, (r.waste, r.obj_id));
                if best.len() > 10 {
                    best.truncate(10);
                }
            }
            let sample_ids: Vec<u64> = best.iter().map(|&(_, id)| id).collect();
            (
                store.as_ref().map_or(records.len() as u64, |s| s.len()),
                total_bad_bytes,
                categories,
                groups,
                sample_ids,
            )
        };

        let is_primitive_array = |name: &str| {
            matches!(
                name,
                "boolean[]" | "byte[]" | "char[]" | "short[]" | "int[]" | "long[]"
                    | "float[]" | "double[]"
            )
        };
        let total_prim_arrays: u64 = self
            .stats
            .per_class
            .iter()
            .filter(|pc| is_primitive_array(&pc.class_name))
            .map(|pc| pc.array_live_count.load(Relaxed) as u64)
            .sum();

        let indicator = if total_bad > 0 && calc_percent(total_bad_bytes, total_bytes) >= 1.0 {
            severity_low(&format!(
                "Overhead {:.1}% ({})",
                calc_percent(total_bad_bytes, total_bytes),
                format_bytes(total_bad_bytes)
            ))
        } else if total_bad > 0 {
            severity_low(&format!("{}", format_bytes(total_bad_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Bad Primitive Arrays", false, &indicator);

        if total_bad > 0 {
            html.begin_table(
                true,
                0,
                &["Total primitive arrays", "Bad primitive arrays", "Overhead"],
            );
            html.append_table_row(&[
                &format_count(total_prim_arrays),
                &format_count(total_bad),
                &format_with_percent(total_bad_bytes, total_bytes),
            ]);
            html.end_table();


            html.begin_table(true, 0, &["Problem type", "# arrays", "Size"]);
            for (kind, count, bytes) in &categories {
                html.append_table_row(&[
                    ss_stats::problem_types::prim_array_problem_name(*kind),
                    &format_count(*count),
                    &format_with_percent(*bytes, total_bytes),
                ]);
            }
            html.end_table();


            html.append_raw(
                "<details><summary>Per-class breakdown</summary><div class=\"scroll-wrap\">",
            );
            html.begin_table(true, 0, &["Overhead", "Problem", "# objects", "Class name"]);
            for ((class_idx, kind), (count, bytes)) in groups.iter().take(50) {
                html.append_table_row(&[
                    &format_with_percent(*bytes, total_bytes),
                    ss_stats::problem_types::prim_array_problem_name(*kind),
                    &format_count(*count),
                    &self.get_class_display(*class_idx),
                ]);
            }
            html.end_table();
            html.append_raw("</div></details>");

            html.append_raw("<details><summary><b>What can I do to fix?</b></summary>\n<div>\n");
            html.append_raw("<table class=\"dt\"><tr><td class=\"TAL\"><font class=\"Bigger\">\
                Empty primitive arrays waste memory due to object headers. Use null references or shared empty arrays.<br>\
                Trailing-zero arrays (where more than the configured share of elements at the end are zero) indicate \
                that the array was allocated larger than needed. Consider sizing arrays more precisely, or using data \
                structures that compress or sparse-encode the data.<br>\
                Arrays with many zero elements or unused high bits can often be replaced by sparse structures or \
                smaller element types.\
                </font></td></tr></table>\n");
            html.append_raw("</div>\n</details>\n");

            // Per-instance reference chains for the largest problem arrays.
            self.render_instance_ref_chains(
                html,
                "Per-instance reference chains for sample bad primitive arrays",
                total_bad_bytes,
                total_bytes,
                &sample_ids,
                Some("Reference Chains for Bad Primitive Arrays"),
            );
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No bad primitive arrays found.", ""]);
            html.end_table();
        }
        Ok(())
    }

    fn section_boxed_numbers(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);

        // Collect boxed number classes into owned data to release borrow
        let mut boxed_classes: Vec<(i64, i64, u64, String)> = Vec::new();
        let mut total_overhead: i64 = 0;
        let mut total_boxed_count: u64 = 0;

        for pc in &self.stats.per_class {
            let overhead = pc.boxed_number_overhead.load(Relaxed);
            if overhead > 0 {
                let count = pc.instance_count.load(Relaxed);
                let size = pc.total_instance_bytes.load(Relaxed);
                total_overhead += overhead;
                total_boxed_count += count;
                boxed_classes.push((overhead, size, count, pc.class_name.replace('/', ".")));
            }
        }

        let pct = calc_percent(total_overhead, total_bytes);

        let indicator = if total_overhead > 0 && pct >= 2.0 {
            severity_high(&format!("Overhead {:.1}% ({})", pct, format_bytes(total_overhead)))
        } else if total_overhead > 0 && pct >= 0.5 {
            severity_med(&format!("Overhead {:.1}% ({})", pct, format_bytes(total_overhead)))
        } else if total_overhead > 0 {
            severity_low(&format!("Overhead {:.1}% ({})", pct, format_bytes(total_overhead)))
        } else {
            String::new()
        };

        self.begin_section(html, "Boxed Numbers", false, &indicator);

        // Summary
        html.begin_table(true, 5, &["Total boxed objects", "Overhead"]);
        html.append_table_row(&[
            &format_count(total_boxed_count),
            &format_with_percent(total_overhead, total_bytes),
        ]);
        html.end_table();

        // Per-class detail
        if !boxed_classes.is_empty() {
            boxed_classes.sort_by(|a, b| b.0.cmp(&a.0));
            html.append_raw("<details><summary>Top boxed number objects</summary><div class=\"scroll-wrap\">");
            html.begin_table(true, 0, &["Overhead", "Size", "# objects", "Type"]);
            for (overhead, size, count, name) in boxed_classes.iter().take(50) {
                html.append_table_row(&[
                    &format_with_percent(*overhead, total_bytes),
                    &format_bytes(*size),
                    &format_count(*count),
                    name,
                ]);
            }
            html.end_table();
            html.append_raw("</div></details>");

            // Typical reference paths for the top boxed class.
            if let Some((_, _, _, top_name)) = boxed_classes.first() {
                let _ = html.append_raw(&self.render_class_ref_chain_panel(top_name, total_bytes));
            }

            // Per-instance ref chain for sample boxed number instances.
            // Scan traversal_data for top boxed class instances.
            if let Some((_, _, _, top_name)) = boxed_classes.first() {
                let lookup = top_name.replace('.', "/");
                let sample_ids: Vec<u64> = {
                    let guard = self.stats.traversal_data.lock().unwrap();
                    if let Some(data) = guard.as_ref() {
                        if let Some(&class_idx) = self.heap.class_by_name.get(lookup.as_str()) {
                            data.sample_ids_by_class(class_idx, 20)
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                };
                self.render_instance_ref_chains(
                    html,
                    &format!("Per-instance reference chains for sample <t-c>{}</t-c> instances", top_name),
                    total_overhead,
                    total_bytes,
                    &sample_ids,
                    Some("Reference Chains for Boxed Numbers"),
                );
            }
        }
    }

    fn section_duplicate_objects(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);

        self.begin_section(html, "Duplicate Objects", false, "");

        // Find classes with many small instances that may indicate duplication.
        // Exclude: String (covered in Section 8), arrays (covered in Sections 14/15),
        // Class objects, Thread objects, and HashMap nodes.
        let exclude_prefixes = [
            "java/lang/String", "java/lang/Class", "java/lang/Thread",
            "java/util/HashMap$Node", "java/util/LinkedHashMap$Entry",
            "java/util/concurrent/ConcurrentHashMap$Node",
        ];

        let mut candidates: Vec<(&str, u64, i64, i64)> = Vec::new(); // (name, count, bytes, instance_size)

        for pc in &self.stats.per_class {
            let count = pc.instance_count.load(Relaxed);
            let bytes = pc.total_instance_bytes.load(Relaxed);
            let inst_size = pc.instance_size;

            if count < 1000 || inst_size > 64 {
                continue;
            }

            let excluded = exclude_prefixes.iter().any(|p| pc.class_name.starts_with(p));
            if excluded {
                continue;
            }
            // Skip array classes
            if pc.class_name.starts_with('[') || pc.class_name.ends_with("[]") {
                continue;
            }

            candidates.push((&pc.class_name, count, bytes, inst_size as i64));
        }

        if !candidates.is_empty() {
            candidates.sort_by(|a, b| b.2.cmp(&a.2));

            html.begin_table(true, 0, &["# instances", "Instance size", "Total shallow size", "Class name"]);
            for (name, count, bytes, inst_size) in candidates.iter().take(30) {
                html.append_table_row(&[
                    &format_count(*count),
                    &format_bytes(*inst_size),
                    &format_with_percent(*bytes, total_bytes),
                    name,
                ]);
            }
            html.end_table();

            html.append_raw("<br><i>These classes have many small instances (1,000+ and instance size &le; 64 bytes). \
                They may contain duplicate values — further investigation via reference chain analysis is recommended.</i>");

            html.append_raw("<details><summary><b>What can I do to fix?</b></summary>\n<div>\n");
            html.append_raw("<table class=\"dt\"><tr><td class=\"TAL\"><font class=\"Bigger\">\
                If these objects hold identical data, consider:<br>\
                - Using a flyweight/canonical object pattern (similar to String.intern())<br>\
                - Caching frequently-created instances in an Enum or static final set<br>\
                - Using value types (Java 16+ records or Project Valhalla value classes in the future)<br>\
                - Reducing unnecessary object creation in hot loops<br>\
                <br>Note: Full duplicate object detection requires reference chain analysis which is planned for a future release.\
                </font></td></tr></table>\n");
            html.append_raw("</div>\n</details>\n");

            // Typical reference paths for the top candidate class.
            if let Some((name, _, _, _)) = candidates.first() {
                let display = name.replace('/', ".");
                let _ = html.append_raw(&self.render_class_ref_chain_panel(&display, total_bytes));

                // Per-instance ref chain for sample instances of the top candidate class.
                let sample_ids: Vec<u64> = {
                    let guard = self.stats.traversal_data.lock().unwrap();
                    if let Some(data) = guard.as_ref() {
                        let key: &str = name.as_ref();
                        if let Some(&class_idx) = self.heap.class_by_name.get(key) {
                            data.sample_ids_by_class(class_idx, 20)
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                };
                self.render_instance_ref_chains(
                    html,
                    &format!("Per-instance reference chains for sample <t-c>{}</t-c> instances", display),
                    candidates.iter().map(|(_, _, b, _)| *b).sum::<i64>(),
                    total_bytes,
                    &sample_ids,
                    Some("Reference Chains for Duplicate Objects"),
                );
            }
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No high-count small-object classes detected.", ""]);
            html.end_table();
        }
    }

    fn section_dup_prim_arrays(&mut self, html: &mut HtmlStringBuilder) {
        let report = self.stats.dup_prim_arrays.get_report();
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        let pct = calc_percent(report.wasted_bytes, total_bytes);

        let indicator = if report.wasted_bytes > 0 && pct >= 2.0 {
            severity_high(&format!("Overhead {:.1}% ({})", pct, format_bytes(report.wasted_bytes)))
        } else if report.wasted_bytes > 0 {
            severity_low(&format!("Overhead {:.1}% ({})", pct, format_bytes(report.wasted_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Duplicate Primitive Arrays", false, &indicator);

        if report.total_count > 0 {
            html.begin_table(true, 5, &["Total arrays", "Unique values", "Duplicate groups", "Overhead"]);
            html.append_table_row(&[
                &format_count(report.total_count as u64),
                &format_count(report.unique_count as u64),
                &format_count(report.duplicate_group_count as u64),
                &format_with_percent(report.wasted_bytes, total_bytes),
            ]);
            html.end_table();

            if !report.per_class.is_empty() {
                let mut per_class = report.per_class.clone();
                per_class.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));
                html.append_raw("<details><summary>Per-class breakdown</summary><div class=\"scroll-wrap\">");
                html.begin_table(true, 0, &["Overhead", "# objects", "Unique", "Class name"]);
                for entry in per_class.iter().take(50) {
                    html.append_table_row(&[
                        &format_with_percent(entry.wasted_bytes, total_bytes),
                        &format_count(entry.instance_count as u64),
                        &format_count(entry.unique_count as u64),
                        &entry.class_name,
                    ]);
                }
                html.end_table();
                html.append_raw("</div></details>");
            }

            if !report.top_groups.is_empty() {
                html.append_raw("<details><summary>Top duplicate arrays</summary><div class=\"scroll-wrap\">");
                html.begin_table(true, 0, &["Overhead", "# objects", "Value"]);
                for g in report.top_groups.iter().take(100) {
                    let value = if g.class_name == "char[]" {
                        format!("{}{{''}}", g.description)
                    } else {
                        format!("{}{{}}", g.description)
                    };
                    html.append_table_row(&[
                        &format_with_percent(g.wasted_bytes, total_bytes),
                        &format_count(g.count as u64),
                        &value,
                    ]);
                }
                html.end_table();
                html.append_raw("</div></details>");
            }

            // Per-instance reference chains for the sample duplicate arrays.
            // Do not use the class-wide chain panel here: it extrapolates all
            // byte[] chains and can produce impossible percentages.
            let mut sample_ids: Vec<u64> = Vec::new();
            for g in report.top_groups.iter().take(10) {
                for &id in g.sample_obj_ids.iter().take(2) {
                    sample_ids.push(id);
                }
            }
            self.render_instance_ref_chains(
                html,
                "Per-instance reference chains for sample duplicate primitive arrays",
                report.wasted_bytes,
                total_bytes,
                &sample_ids,
                Some("Reference Chains for Duplicate Primitive Arrays"),
            );
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No duplicate primitive arrays detected", ""]);
            html.end_table();
        }
    }

    fn section_dup_obj_arrays(&mut self, html: &mut HtmlStringBuilder) {
        let report = self.stats.dup_obj_arrays.get_report();
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        let pct = calc_percent(report.wasted_bytes, total_bytes);

        let indicator = if report.wasted_bytes > 0 && pct >= 2.0 {
            severity_high(&format!("Overhead {:.1}% ({})", pct, format_bytes(report.wasted_bytes)))
        } else if report.wasted_bytes > 0 {
            severity_low(&format!("Overhead {:.1}% ({})", pct, format_bytes(report.wasted_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Duplicate Object Arrays", false, &indicator);

        // Java's buildDupReport returns null when no class clears the 0.1%
        // threshold; renderDuplicateObjectsSection then emits the header only.
        if !report.report_present {
            return;
        }

        if report.total_count > 0 {
            html.begin_table(true, 5, &["Total arrays", "Unique values", "Duplicate groups", "Overhead"]);
            html.append_table_row(&[
                &format_count(report.total_count as u64),
                &format_count(report.unique_count as u64),
                &format_count(report.duplicate_group_count as u64),
                &format_with_percent(report.wasted_bytes, total_bytes),
            ]);
            html.end_table();

            if !report.per_class.is_empty() {
                let mut per_class = report.per_class.clone();
                per_class.sort_by(|a, b| b.wasted_bytes.cmp(&a.wasted_bytes));
                html.append_raw("<details><summary>Per-class breakdown</summary><div class=\"scroll-wrap\">");
                html.begin_table(true, 0, &["Overhead", "# objects", "Unique", "Class name"]);
                for entry in per_class.iter().take(50) {
                    html.append_table_row(&[
                        &format_with_percent(entry.wasted_bytes, total_bytes),
                        &format_count(entry.instance_count as u64),
                        &format_count(entry.unique_count as u64),
                        &entry.class_name,
                    ]);
                }
                html.end_table();
                html.append_raw("</div></details>");

                // Typical reference paths for the top duplicate class.
                if let Some(top) = per_class.first() {
                    let _ = html.append_raw(&self.render_class_ref_chain_panel(&top.class_name, total_bytes));
                }
            }

            // Per-instance reference chains for sample duplicate object arrays.
            let mut sample_ids: Vec<u64> = Vec::new();
            for g in report.top_groups.iter().take(10) {
                for &id in g.sample_obj_ids.iter().take(2) {
                    sample_ids.push(id);
                }
            }
            self.render_instance_ref_chains(
                html,
                "Per-instance reference chains for sample duplicate object arrays",
                report.wasted_bytes,
                total_bytes,
                &sample_ids,
                None,
            );
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No duplicate object arrays detected", ""]);
            html.end_table();
        }
    }

    fn section_dup_lists(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        self.begin_section(html, "Duplicate Lists", false, "");

        // List-like classes we care about.
        let list_classes = [
            "java/util/ArrayList",
            "java/util/LinkedList",
            "java/util/Collections$UnmodifiableRandomAccessList",
            "java/util/Collections$SingletonList",
            "java/util/Arrays$ArrayList",
        ];

        let mut entries: Vec<(String, u64, i64)> = Vec::new();
        for cls in list_classes {
            if let Some(&idx) = self.heap.class_by_name.get(cls) {
                if let Some(pc) = self.stats.get_per_class(idx) {
                    let count = pc.instance_count.load(Relaxed);
                    let bytes = pc.total_instance_bytes.load(Relaxed);
                    if count > 0 {
                        entries.push((pc.class_name.replace('/', "."), count, bytes));
                    }
                }
            }
        }

        if entries.is_empty() {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No list instances found in this heap dump.", ""]);
            html.end_table();
            return;
        }

        // Summary table
        let total_count: u64 = entries.iter().map(|(_, c, _)| *c).sum();
        let total_size: i64 = entries.iter().map(|(_, _, b)| *b).sum();
        let pct = calc_percent(total_size, total_bytes);
        let indicator = if pct >= 2.0 {
            severity_med(&format!("Overhead {:.1}% ({})", pct, format_bytes(total_size)))
        } else if pct >= 0.5 {
            severity_low(&format!("Overhead {:.1}% ({})", pct, format_bytes(total_size)))
        } else {
            String::new()
        };
        let _ = indicator; // begin_section was already called without indicator

        html.begin_table(true, 5, &["Total list objects", "Total size"]);
        html.append_table_row(&[
            &format_count(total_count),
            &format_with_percent(total_size, total_bytes),
        ]);
        html.end_table();

        // Per-class breakdown
        entries.sort_by(|a, b| b.2.cmp(&a.2));
        html.append_raw("<details><summary>Per-class breakdown</summary><div class=\"scroll-wrap\">");
        html.begin_table(true, 0, &["# objects", "Size", "Class name"]);
        for (name, count, bytes) in entries.iter().take(50) {
            html.append_table_row(&[
                &format_count(*count),
                &format_with_percent(*bytes, total_bytes),
                name,
            ]);
        }
        html.end_table();
        html.append_raw("</div></details>");

        html.append_raw(&format!(
            "<br><i>Note: Full duplicate list detection (lists with identical element sequences) \
             requires container walker integration. The reference paths below show where the \
             most-common list class lives.</i><br>"
        ));

        // Typical reference paths + per-instance chains for top list class.
        if let Some((top_name, _, _)) = entries.first() {
            let lookup = top_name.replace('.', "/");
            let _ = html.append_raw(&self.render_class_ref_chain_panel(top_name, total_bytes));

            // Per-instance ref chain for sample list instances.
            let sample_ids: Vec<u64> = {
                let guard = self.stats.traversal_data.lock().unwrap();
                if let Some(data) = guard.as_ref() {
                    if let Some(&class_idx) = self.heap.class_by_name.get(lookup.as_str()) {
                        data.sample_ids_by_class(class_idx, 20)
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            self.render_instance_ref_chains(
                html,
                &format!("Per-instance reference chains for sample <t-c>{}</t-c> instances", top_name),
                total_size,
                total_bytes,
                &sample_ids,
                None,
            );
        }
    }

    fn section_weak_hashmaps(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        self.begin_section(
            html,
            "WeakHashMaps with hard references from values to keys",
            false,
            "",
        );

        // Count WeakHashMap instances
        let whm_class_idx = self.heap.class_by_name.get("java/util/WeakHashMap").copied();
        let whm_count = whm_class_idx
            .and_then(|idx| self.stats.get_per_class(idx))
            .map(|pc| pc.instance_count.load(Relaxed))
            .unwrap_or(0);

        if whm_count > 0 {
            html.begin_table(true, 5, &["WeakHashMap instances", "Count"]);
            html.append_table_row(&["java.util.WeakHashMap", &format_count(whm_count)]);
            html.end_table();

            html.append_raw(&format!(
                "<br><i>A WeakHashMap holds its keys via WeakReference, so keys can be \
                 garbage-collected when no longer strongly reachable. However, if a value \
                 strongly references its key (directly or transitively), the key can never \
                 be GC'd — effectively turning the WeakHashMap into a leaky HashMap.</i><br><br>"
            ));

            // Per-instance ref chain for sample WeakHashMap instances.
            // Scan traversal_data for WeakHashMap instances.
            let sample_ids: Vec<u64> = {
                let guard = self.stats.traversal_data.lock().unwrap();
                if let Some(data) = guard.as_ref() {
                    if let Some(class_idx) = whm_class_idx {
                        data.sample_ids_by_class(class_idx, 20)
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            self.render_instance_ref_chains(
                html,
                "Reference chains holding WeakHashMap instances",
                whm_count as i64 * 64, // rough size estimate
                total_bytes,
                &sample_ids,
                Some("Reference Chains for bad WeakHashMaps"),
            );
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No WeakHashMap instances found in this heap dump.", ""]);
            html.end_table();
        }
    }

    fn section_direct_byte_buffers(&mut self, html: &mut HtmlStringBuilder) {
        let report = self.stats.direct_buffers.get_report();
        let total_heap = self.stats.total_shallow_size.load(Relaxed).max(1);

        let indicator = if report.total_capacity > 0 {
            let pct = calc_percent(report.total_capacity, total_heap);
            if pct >= 5.0 {
                severity_high(&format!("Off-heap {} ({:.1}% of heap)", format_bytes(report.total_capacity), pct))
            } else if pct >= 2.0 {
                severity_med(&format!("Off-heap {} ({:.1}% of heap)", format_bytes(report.total_capacity), pct))
            } else {
                severity_low(&format!("Off-heap {}", format_bytes(report.total_capacity)))
            }
        } else {
            String::new()
        };

        self.begin_section(
            html,
            "Off-heap memory used by java.nio.DirectByteBuffers",
            false,
            &indicator,
        );

        if report.total_count > 0 {
            // Split reachable/unreachable via BFS `visited`. Buffers not
            // reachable from any GC root are "unreachable" (garbage) and are
            // the actionable subset Java highlights (e.g. 94,786KB / 53.2%).
            let unreachable_capacity = self.compute_unreachable_dbb_capacity(&report);
            let reachable_capacity = report.total_capacity - unreachable_capacity;

            html.begin_table(true, 0, &["# objects", "Total capacity", "Total size"]);
            html.append_table_row(&[
                &format_count(report.total_count as u64),
                &format_with_percent(reachable_capacity, total_heap),
                &format_with_percent(report.total_capacity, total_heap),
            ]);
            html.end_table();

            // Unreachable (garbage) DirectByteBuffers — the easy-to-fix subset.
            if unreachable_capacity > 0 {
                let pct = calc_percent(unreachable_capacity, total_heap);
                let sev = if pct >= 5.0 { severity_high } else if pct >= 2.0 { severity_med } else { severity_low };
                html.append_raw(&format!(
                    "<details><summary>{}</summary><div><table class=\"dt\"><br>\
                     <tr class=\"C_02\"><td class=\"TAL\">DirectByteBuffer objects that are unreachable \
                     (garbage) but still hold native (off-heap) memory: <b>{}</b> &nbsp;({:.1}% of used heap)</td></tr>\
                     </table></div></details><br>",
                    sev(&format!("unreachable DirectByteBuffers holding off-heap memory. Overhead: {} ({:.1}% of heap).", format_bytes(unreachable_capacity), pct)),
                    format_bytes(unreachable_capacity), pct,
                ));
            }

            if !report.per_class.is_empty() {
                html.append_raw("<details><summary>Per-type breakdown</summary><div class=\"scroll-wrap\">");
                html.begin_table(true, 0, &["Count", "Off-heap capacity", "Type"]);
                for entry in &report.per_class {
                    html.append_table_row(&[
                        &format_count(entry.count as u64),
                        &format_bytes(entry.total_capacity),
                        &entry.class_name,
                    ]);
                }
                html.end_table();
                html.append_raw("</div></details>");

                // Typical reference paths for the top DirectBuffer subclass.
                if let Some(top) = report.per_class.first() {
                    let _ = html.append_raw(&self.render_class_ref_chain_panel(&top.class_name, total_heap));
                }
            }

            // Per-instance reference chains for top DirectByteBuffers.
            self.render_instance_ref_chains(
                html,
                "Per-instance reference chains for largest DirectByteBuffers",
                report.total_capacity,
                total_heap,
                &report.sample_obj_ids,
                Some("Reference Chains for java.nio.DirectByteBuffers"),
            );

            // Easy-to-fix advice panel, mirroring Java which renders the
            // unreachable-DirectByteBuffer advice both in the MOST IMPORTANT
            // ISSUES overview and again here in Section-18.
            if let Some(panel) = self.render_direct_buffer_advice(total_heap) {
                html.append_raw(&panel);
            }
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No DirectByteBuffer instances found", ""]);
            html.end_table();
        }
    }

    fn section_humongous_objects(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed);
        let jvm_info = self.stats.jvm_info.get();
        let g1_region_size = jvm_info.g1_region_size.max(1024 * 1024);
        let _half_region = g1_region_size / 2;

        // Collect per-class humongous stats into owned data to avoid borrow issues
        struct HumongousClass {
            class_index: u32,
            class_name: String,
            count: u32,
            size: i64,
            overhead: i64,
        }
        let mut humongous_classes: Vec<HumongousClass> = Vec::new();
        let mut total_count: u64 = 0;
        let mut total_size: i64 = 0;
        let mut total_overhead: i64 = 0;

        for pc in &self.stats.per_class {
            let count = pc.humongous_count.load(Relaxed);
            let bytes = pc.humongous_bytes.load(Relaxed);
            let overhead = pc.humongous_overhead.load(Relaxed);
            if count > 0 {
                total_count += count as u64;
                total_size += bytes;
                total_overhead += overhead;
                humongous_classes.push(HumongousClass {
                    class_index: pc.class_index,
                    class_name: pc.class_name.clone(),
                    count,
                    size: bytes,
                    overhead,
                });
            }
        }

        humongous_classes.sort_by(|a, b| b.size.cmp(&a.size));

        let overhead_pct = calc_percent(total_overhead, total_bytes);
        let indicator = if total_overhead > 0 {
            let color = if overhead_pct >= 6.0 { "OvhdColorHigh" } else if overhead_pct >= 2.0 { "OvhdColorMed" } else { "OvhdColorLow" };
            format!(
                "<font class=\"{}\">&nbsp;Overhead {:.1}%&nbsp ( {} )&nbsp;</font>",
                color,
                overhead_pct,
                format_bytes(total_overhead)
            )
        } else {
            String::new()
        };

        self.begin_section(html, "Humongous (bigger than 1MB) objects.", false, &indicator);

        // G1 GC note
        let gc_type = &jvm_info.gc_type;
        let is_g1_or_similar = gc_type == "G1" || gc_type == "Shenandoah" || gc_type == "ZGC";
        if is_g1_or_similar {
            html.append_raw(&format!(
                "The JVM that ran this app quite possibly uses {} Garbage Collector<br><br>",
                if gc_type == "G1" { "G1" } else if gc_type == "Shenandoah" { "Shenandoah" } else { "ZGC" }
            ));
        }
        html.append_raw(
            "Note that the overhead (wasted memory) below is IN ADDITION to the Used Memory<br>\
             which we give throughout this report. That is, Humongous objects use more memory<br>\
             than their useful workload requires.<br>"
        );

        // Summary table: Total humongous objects, Total size, Total overhead
        html.begin_table_with_tooltips(true, 5, &[
            ("Total humongous objects", "How many humongous objects are present in this dump."),
            ("Total size", "How much memory is used by all humongous objects."),
            ("Total overhead", "How much memory is wasted in underutilized humongous heap regions."),
        ]);
        html.append_table_row(&[
            &format_count(total_count),
            &format_with_percent(total_size, total_bytes.max(1)),
            &format_with_percent(total_overhead, total_bytes.max(1)),
        ]);
        html.end_table();

        // Per-class breakdown table
        if !humongous_classes.is_empty() {
            html.append_raw("<br>Top potentially humongous objects:<br>");
            html.begin_table_with_tooltips(true, 0, &[
                ("Humongous obj overhead", "How much memory is wasted by all the humongous objects of the given type."),
                ("Humongous obj size", "How much memory all the humongous objects of the given type take."),
                ("# humongous objects", "Number of humongous objects of the given type."),
                ("Type", ""),
            ]);
            for hc in &humongous_classes {
                let name = hc.class_name.replace('/', ".");
                html.append_table_row(&[
                    &format_with_percent(hc.overhead, total_bytes.max(1)),
                    &format_with_percent(hc.size, total_bytes.max(1)),
                    &format_count(hc.count as u64),
                    &format!("<t-c>{}</t-c>", name),
                ]);
            }
            html.end_table();
        }

        // Reference chains for humongous objects
        if total_count > 0 {
            html.append_raw("<br><font class=\"SecHdr\">Reference Chains for humongous objects</font><br><br>");

            // Use humongous-specific ref chains (not class-level)
            let ref_chains = self.stats.ref_chains.lock().unwrap();
            let chains = &ref_chains.humongous_chains;

            if !chains.expensive_fields.is_empty() || !chains.full_chains.is_empty() {
                render_humongous_ref_chains(chains, html, total_bytes);
            } else {
                html.append_raw(
                    "<details><summary>Expensive data fields</summary><br>\
                     <div class=\"scroll-wrap\">Reference chain analysis not yet available for humongous objects.</div>\
                     </details><br>"
                );
            }
        }
    }

    fn section_heap_config(&mut self, html: &mut HtmlStringBuilder) {
        self.begin_section(html, "Heap Size Configuration", false, "");
        html.begin_table(true, 10, &["", ""]);
        html.append_table_row(&[
            "Identifier size",
            &format!("{} bytes", self.heap.identifier_size),
        ]);
        html.append_table_row(&[
            "Object header size",
            &format!("{} bytes", self.heap.header_size),
        ]);
        html.append_table_row(&[
            "Object alignment",
            &format!("{} bytes", self.heap.alignment),
        ]);
        html.append_table_row(&[
            "Total heap dump size",
            &format_bytes(self.stats.total_shallow_size.load(Relaxed)),
        ]);
        html.end_table();
    }

    fn section_long_ref_chains(&mut self, html: &mut HtmlStringBuilder) {
        // Find objects whose parent[] chain to root exceeds the threshold.
        let threshold: u32 = 1000;
        let top_n: usize = 20;

        // Collect all data while holding the lock; render after release.
        struct ChainRow {
            depth: u32,
            obj_id: u64,
            class_name: String,
            path: Vec<(u64, String, String)>, // (parent_obj_id, parent_class, field)
            reached_root: bool,
        }

        let rows: Vec<ChainRow> = {
            let guard = self.stats.traversal_data.lock().unwrap();
            let data = match guard.as_ref() {
                Some(d) => d,
                None => {
                    drop(guard);
                    self.begin_section(
                        html,
                        "Very Long (Over 1000 Elements) Reference Chains",
                        false,
                        "",
                    );
                    html.begin_table(true, 5, &["", ""]);
                    html.append_table_row(&[
                        "Reference chain analysis requires BFS traversal data.",
                        "Run the analyzer with the standard pipeline.",
                    ]);
                    html.end_table();
                    return;
                }
            };

            // Walk parent[] for every visited object, compute chain depth.
            let mut depths = DepthMemo::new(data.object_count);
            let mut visited_chain: Vec<u32> = Vec::with_capacity(64);
            // The section renders long_chains.len() rows at most, so the collector
            // keeps that many entries. Recording one entry per node of every deep
            // chain instead costs a megabyte per ten thousand nodes, and a heap full
            // of caches and linked lists has hundreds of millions of them.
            let mut long_chains: Vec<(u32, u32)> = Vec::with_capacity(top_n); // (depth, obj_idx)

            for idx in 0..data.object_count {
                if !data.visited.get(idx) || depths.get(idx) > 0 {
                    continue;
                }
                visited_chain.clear();
                let mut current = idx;
                loop {
                    if depths.get(current) > 0 {
                        // Found a known depth — backfill.
                        let mut d = depths.get(current);
                        for &node in visited_chain.iter().rev() {
                            d += 1;
                            depths.set(node, d);
                            if d >= threshold {
                                record_long_chain(&mut long_chains, d, node, top_n);
                            }
                        }
                        break;
                    }
                    let parent = data.parent.parent_of(current).0;
                    if parent == u32::MAX {
                        depths.set(current, 1);
                        let mut d = 1u32;
                        for &node in visited_chain.iter().rev() {
                            d += 1;
                            depths.set(node, d);
                            if d >= threshold {
                                record_long_chain(&mut long_chains, d, node, top_n);
                            }
                        }
                        break;
                    }
                    visited_chain.push(current);
                    current = parent;
                }
            }

            long_chains.sort_by(|a, b| b.0.cmp(&a.0));
            long_chains.truncate(top_n);

            long_chains.iter().filter_map(|&(depth, obj_idx)| {
                let obj = data.object_at(obj_idx)?;
                let class_name = if (obj.class_idx as usize) < self.heap.classes.len() {
                    self.heap.classes[obj.class_idx as usize].name.replace('/', ".")
                } else {
                    "unknown".to_string()
                };
                // Walk path — cap at 50 hops for display.
                let max_display: u32 = 50;
                let mut path: Vec<(u64, String, String)> = Vec::new();
                let mut current = obj_idx;
                let mut hops = 0u32;
                let mut reached_root = true;
                loop {
                    let parent = data.parent.parent_of(current).0;
                    if parent == u32::MAX {
                        break;
                    }
                    let parent_obj = match data.object_at(parent) {
                        Some(o) => o,
                        None => break,
                    };
                    let parent_class = if (parent_obj.class_idx as usize) < self.heap.classes.len() {
                        self.heap.classes[parent_obj.class_idx as usize].name.replace('/', ".")
                    } else {
                        "unknown".to_string()
                    };
                    let edge = data.parent.parent_of(current).1;
                    let field = data.field_pool.get(edge).to_string();
                    path.push((parent_obj.obj_id, parent_class, field));
                    current = parent;
                    hops += 1;
                    if hops >= max_display {
                        reached_root = false;
                        break;
                    }
                }
                Some(ChainRow {
                    depth,
                    obj_id: obj.obj_id,
                    class_name,
                    path,
                    reached_root,
                })
            }).collect()
        };

        let indicator = if rows.is_empty() {
            String::new()
        } else {
            severity_med(&format!("{} chains > {} hops", rows.len(), threshold))
        };

        self.begin_section(
            html,
            "Very Long (Over 1000 Elements) Reference Chains",
            false,
            &indicator,
        );

        if rows.is_empty() {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&[
                &format!("No reference chains longer than {} elements found.", threshold),
                "",
            ]);
            html.end_table();
        } else {
            html.begin_table(true, 0, &["Chain length", "Class", "Object", "Reference path"]);
            for row in &rows {
                let mut path_html = String::new();
                for (i, (_, cls, fld)) in row.path.iter().enumerate() {
                    let tab = (3 + i.min(7)) as u32;
                    path_html.push_str(&format!(
                        "<rc_tab{}>&nwarr;<t-c>{}</t-c><t-f>.{}</t-f><br></rc_tab{}>",
                        tab, cls, fld, tab,
                    ));
                }
                if !row.reached_root {
                    path_html.push_str(&format!(
                        "<rc_tab8>... and {} more hops to root<br></rc_tab8>",
                        row.depth.saturating_sub(row.path.len() as u32),
                    ));
                }
                html.append_table_row(&[
                    &format_count(row.depth as u64),
                    &format!("<t-c>{}</t-c>", row.class_name),
                    &format!("<t-c>{}</t-c>@0x{:x}", row.class_name, row.obj_id),
                    &path_html,
                ]);
            }
            html.end_table();
        }
    }

    fn get_class_display(&self, class_idx: u32) -> String {
        if (class_idx as usize) < self.heap.classes.len() {
            self.heap.classes[class_idx as usize].name.replace('/', ".")
        } else {
            "unknown".to_string()
        }
    }

    fn section_thread_stacks(&mut self, html: &mut HtmlStringBuilder) {
        let thread_info: Vec<_> = self.stats.thread_info.lock().unwrap().clone();

        let thread_count = self.heap.stack_traces.len();
        let indicator = if thread_count > 0 {
            format!(". (number of threads<font class=\"OvhdColorLow\">&nbsp;{}&nbsp;</font>)", thread_count)
        } else {
            String::new()
        };
        self.begin_section(html, "Thread stacks", false, &indicator);

        if thread_info.is_empty() && self.heap.stack_traces.is_empty() {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No thread stack information available in this heap dump.", ""]);
            html.end_table();
            return;
        }

        // Summary table
        let mut thread_roots = 0u64;
        let mut stack_roots = 0u64;
        for root in &self.heap.roots {
            match root.root_type {
                ss_heap_model::root::RootType::ThreadObj => thread_roots += 1,
                ss_heap_model::root::RootType::JavaStackFrame | ss_heap_model::root::RootType::NativeStack => stack_roots += 1,
                _ => {}
            }
        }

        html.begin_table(true, 0, &["Thread data", "Count"]);
        html.append_table_row(&["Thread objects", &format_count(thread_roots)]);
        html.append_table_row(&["Stack traces", &format_count(thread_count as u64)]);
        html.append_table_row(&["Stack frame roots", &format_count(stack_roots)]);
        html.end_table();

        // Build serial -> (name, daemon) lookup
        let serials_with_names: Vec<(u32, String, bool)> = thread_info.iter()
            .map(|ti| (ti.thread_serial, ti.name.clone(), ti.is_daemon))
            .collect();

        if !self.heap.stack_traces.is_empty() {
            // Determine the actual OOM thread serial (same logic as section_oom_thread)
            let idle_methods: &[&str] = &["wait", "park", "parkNanos", "parkUntil", "sleep", "yield", "waitForReferencePendingList"];
            let mut oom_serial: Option<u32> = None;
            let mut oom_depth: usize = 0;
            let mut trace_serials: Vec<u32> = self.heap.stack_traces.keys().copied().collect();
            trace_serials.sort_unstable();
            for serial in trace_serials {
                let st = &self.heap.stack_traces[&serial];
                if let Some(first_frame) = st.frames.first() {
                    if first_frame.line_number == -3 {
                        let is_idle = idle_methods.iter().any(|m| first_frame.method_name == *m);
                        if !is_idle && st.frames.len() > oom_depth {
                            oom_serial = Some(serial);
                            oom_depth = st.frames.len();
                        }
                    }
                }
            }

            // Sort threads: OOM thread first, then by name
            let mut sorted_serials: Vec<u32> = self.heap.stack_traces.keys().copied().collect();
            sorted_serials.sort_by_key(|&serial| {
                let is_oom = oom_serial == Some(serial);
                let name = serials_with_names.iter()
                    .find(|(s, _, _)| *s == serial)
                    .map(|(_, n, _)| n.as_str())
                    .unwrap_or("");
                (if is_oom { 0 } else { 1 }, name.to_string(), serial)
            });

            for serial in &sorted_serials {
                if let Some(st) = self.heap.stack_traces.get(serial) {
                    let name = serials_with_names.iter()
                        .find(|(s, _, _)| *s == *serial)
                        .map(|(_, n, d)| format!("\"{}\"{}", n, if *d { " (daemon)" } else { "" }))
                        .unwrap_or_else(|| format!("Thread serial {}", serial));
                    let is_oom = oom_serial == Some(*serial);

                    html.append_raw("<details>");
                    html.append_raw(&format!(
                        "<summary><b>{}{}</b> (serial {})</summary><br>",
                        if is_oom { "[OOM] " } else { "" },
                        name,
                        serial
                    ));

                    if st.frames.is_empty() {
                        html.append_raw("&nbsp;&nbsp;(no stack frames)<br>");
                    } else {
                        for frame in &st.frames {
                            let display_class = if frame.class_name.is_empty() || frame.class_name == "unknown" {
                                "null".to_string()
                            } else {
                                frame.class_name.replace('/', ".")
                            };
                            let line_info = match frame.line_number {
                                -1 => format!("unknown, ID={}, line not available", frame.frame_id),
                                -2 => format!("compiled method, ID={}", frame.frame_id),
                                -3 => "Native method".to_string(),
                                n => format!("{}:{}, ID={}", frame.source_file, n, frame.frame_id),
                            };
                            html.append_raw(&format!(
                                "<t-c>{}</t-c>.<t-m>{}</t-m>({})<br>",
                                display_class,
                                frame.method_name,
                                line_info,
                            ));
                        }
                    }
                    html.append_raw("</details>");
                }
            }
        }
    }

    fn section_system_properties(&mut self, html: &mut HtmlStringBuilder) {
        let info = self.stats.jvm_info.get();

        self.begin_section(html, "System Properties", false, "");

        if info.system_properties.is_empty() {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&[
                "System properties could not be extracted from this heap dump.",
                "",
            ]);
            html.end_table();
        } else {
            html.begin_table(true, 0, &["Property", "Value"]);
            for (key, value) in &info.system_properties {
                let escaped = value.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
                html.append_table_row(&[key, &escaped]);
            }
            html.end_table();
        }
    }

    fn section_jvm_flags(&mut self, html: &mut HtmlStringBuilder) {
        let info = self.stats.jvm_info.get();

        self.begin_section(html, "JVM Flags", false, "");

        if info.jvm_flags_raw.is_empty() && info.gc_type == "Unknown" {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&[
                "JVM flags not found in this heap dump. To include them, set system property 'java.vm.inputarguments' at application startup.",
                "",
            ]);
            html.end_table();
        } else {
            html.begin_table(true, 0, &["Property", "Value"]);
            if !info.gc_type.is_empty() && info.gc_type != "Unknown" {
                html.append_table_row(&["GC Type", &info.gc_type]);
            }
            html.append_table_row(&["Address Size", &format!("{}-bit", info.address_size)]);
            html.append_table_row(&["ID Size", &format!("{} bytes", info.id_size)]);
            if info.xmx > 0 {
                html.append_table_row(&["-Xmx", &format_bytes(info.xmx)]);
            }
            if info.xms > 0 {
                html.append_table_row(&["-Xms", &format_bytes(info.xms)]);
            }
            html.end_table();

            if !info.jvm_flags_raw.is_empty() {
                // Categorize flags by type for clearer presentation.
                let parsed = parse_jvm_flags(&info.jvm_flags_raw);

                // Heap-related flags
                if !parsed.heap_flags.is_empty() {
                    html.append_raw("<details><summary>Heap configuration flags</summary><div class=\"scroll-wrap\">");
                    html.begin_table(true, 0, &["Flag", "Value", "Description"]);
                    for (flag, value, desc) in &parsed.heap_flags {
                        html.append_table_row(&[
                            &format!("<code>{}</code>", html_escape_simple(flag)),
                            &html_escape_simple(value),
                            desc,
                        ]);
                    }
                    html.end_table();
                    html.append_raw("</div></details>");
                }

                // GC-related flags
                if !parsed.gc_flags.is_empty() {
                    html.append_raw("<details><summary>Garbage collection flags</summary><div class=\"scroll-wrap\">");
                    html.begin_table(true, 0, &["Flag", "Value", "Description"]);
                    for (flag, value, desc) in &parsed.gc_flags {
                        html.append_table_row(&[
                            &format!("<code>{}</code>", html_escape_simple(flag)),
                            &html_escape_simple(value),
                            desc,
                        ]);
                    }
                    html.end_table();
                    html.append_raw("</div></details>");
                }

                // Performance/JIT flags
                if !parsed.perf_flags.is_empty() {
                    html.append_raw("<details><summary>Performance / JIT flags</summary><div class=\"scroll-wrap\">");
                    html.begin_table(true, 0, &["Flag", "Value", "Description"]);
                    for (flag, value, desc) in &parsed.perf_flags {
                        html.append_table_row(&[
                            &format!("<code>{}</code>", html_escape_simple(flag)),
                            &html_escape_simple(value),
                            desc,
                        ]);
                    }
                    html.end_table();
                    html.append_raw("</div></details>");
                }

                // Other flags
                if !parsed.other_flags.is_empty() {
                    html.append_raw("<details><summary>Other flags</summary><div class=\"scroll-wrap\">");
                    html.begin_table(true, 0, &["Flag", "Value"]);
                    for (flag, value) in &parsed.other_flags {
                        html.append_table_row(&[
                            &format!("<code>{}</code>", html_escape_simple(flag)),
                            &html_escape_simple(value),
                        ]);
                    }
                    html.end_table();
                    html.append_raw("</div></details>");
                }

                // Full raw dump
                html.append_raw("<details><summary>All flags (raw)</summary><div class=\"scroll-wrap\">");
                html.begin_table(true, 0, &["Flags"]);
                for flag in info.jvm_flags_raw.split_whitespace() {
                    let escaped = flag.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
                    html.append_table_row(&[&escaped]);
                }
                html.end_table();
                html.append_raw("</div></details>");
            }
        }
    }

    /// Sum the off-heap capacity of DirectByteBuffers that are NOT reachable
    /// from any GC root (i.e. garbage), using BFS `visited`. Returns 0 when
    /// traversal data is unavailable. This is the "unreachable DBB" subset
    /// Java highlights as an easy-to-fix problem.
    fn compute_unreachable_dbb_capacity(
        &self,
        report: &ss_stats::direct_buffer::DirectBufferReport,
    ) -> i64 {
        if report.buffers.is_empty() {
            return 0;
        }
        let guard = self.stats.traversal_data.lock().unwrap();
        let data = match guard.as_ref() {
            Some(d) => d,
            None => return 0,
        };
        // Build obj_id -> visited lookup. `visited` is parallel to `objects`.
        // Only buffers whose obj_id appears in objects AND is visited count as
        // reachable; anything absent or not visited is unreachable.
        report
            .buffers
            .iter()
            .filter(|b| !data.is_live_id(b.obj_id))
            .map(|b| b.capacity)
            .sum()
    }

    /// Render the unreachable-DirectByteBuffer easy-to-fix advice as a panel,
    /// for embedding in Section-18 (mirrors Java, which shows this advice both
    /// in the MOST IMPORTANT ISSUES overview and again inside Section-18).
    /// Returns `None` when the advice does not fire.
    fn render_direct_buffer_advice(&self, total_bytes: i64) -> Option<String> {
        let problems = ProblemReportSet::from_stats(&self.stats, &self.heap);
        let ctx = AdvisorContext {
            stats: &self.stats,
            heap: &self.heap,
            problems: &problems,
        };
        let registry = AdvisorRegistry::new();
        let advice = registry.run_all(&ctx).into_iter().find(|a| {
            a.category == AdviceCategory::EasyToFix
                && a.section_id == "Section-18"
        })?;
        Some(render_advice_panel(&advice, total_bytes))
    }

    /// Generate the "MOST IMPORTANT ISSUES" advisor summary section.
    fn generate_advisor_summary(&self) -> String {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        let problems = ProblemReportSet::from_stats(&self.stats, &self.heap);
        let ctx = AdvisorContext {
            stats: &self.stats,
            heap: &self.heap,
            problems: &problems,
        };
        let registry = AdvisorRegistry::new();
        let mut advices = registry.run_all(&ctx);
        advices.sort_by_key(|a| (a.category.order(), a.order_hint));

        let mut summary = String::with_capacity(8192);
        summary.push_str(&format!(
            "<details open id=\"summary-section\">\n<summary>{}</summary><br>\n",
            format_section_header("MOST IMPORTANT ISSUES", Some("#most_important_issues"))
        ));
        summary.push_str("<font class=\"Bigger\">");
        summary.push_str("We found some problems in this heap dump.<br>\n");
        summary.push_str(&format!(
            "For each problem where possible we report {}: how much memory you would save<br>\n",
            bold("overhead")
        ));
        summary.push_str("(in Kbytes and as a percentage of used memory) if you get rid of this problem.<br>\n");
        summary.push_str("<br>\n");

        // Easy-to-fix panel
        let easy: Vec<&Advice> = advices.iter()
            .filter(|a| a.category == AdviceCategory::EasyToFix)
            .collect();
        if !easy.is_empty() {
            summary.push_str(&bigger_font(&bold("Good news! Some problems that we found are easy to fix, see below:")));
            summary.push_str("<br>\n");
            for a in &easy {
                summary.push_str(&render_advice_panel(a, total_bytes));
                summary.push_str("<br>\n");
            }
        }

        summary.push_str(&bigger_font(&bold("All problems:")));
        summary.push_str("<br>\n");
        summary.push_str("<table class=\"dt\">\n");

        // All problems table: include every advice as a summary row
        for a in &advices {
            summary.push_str(&render_advice_table_row(a, total_bytes));
        }

        // Also surface some inline checks that don't yet have dedicated advisors.
        self.append_inline_problem_rows(&mut summary, &problems, total_bytes);

        summary.push_str("</table>\n");
        summary.push_str("</font>\n");
        summary.push_str("</details>\n");
        summary
    }

    /// Append summary rows for problems that don't yet have dedicated advisors
    /// (garbage ratio, humongous objects, dup arrays). These will be migrated
    /// to advisors in later phases.
    fn append_inline_problem_rows(
        &self,
        summary: &mut String,
        problems: &ProblemReportSet,
        total_bytes: i64,
    ) {
        // Garbage ratio
        let garbage_pct = calc_percent(problems.garbage_bytes, total_bytes);
        if garbage_pct > 20.0 {
            summary.push_str(&format!(
                "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Large amount of garbage objects {}</font></td></tr>\n",
                bold("Problem"),
                severity_med(&format!("found - {:.1}% garbage", garbage_pct))
            ));
            summary.push_str(&format!(
                "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                <tr><td class=\"TAL\"><font class=\"Bigger\">Garbage objects occupy {} of heap memory. Consider triggering a GC or investigating object lifecycle.</font></td></tr>\
                </table></div></details></font></td></tr>\n",
                format_with_percent(problems.garbage_bytes, total_bytes.max(1))
            ));
        }

        // Humongous objects (when G1OOM advisor didn't fire, e.g. non-G1 collector)
        if problems.humongous_bytes > 0
            && !problems.gc_type.to_ascii_uppercase().contains("G1")
        {
            summary.push_str(&format!(
                "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Humongous objects (&gt;1MB) found {}</font></td></tr>\n",
                bold("Problem"),
                severity_low(&format!("found - {}", format_bytes(problems.humongous_bytes)))
            ));
            summary.push_str(&format!(
                "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                <tr><td class=\"TAL\"><font class=\"Bigger\">Objects larger than 1MB occupy {} of heap. For full details, go to \
                <button onclick=\"openSection('Section-19');\">Section 19. Humongous objects</button></font></td></tr>\
                </table></div></details></font></td></tr>\n",
                format_with_percent(problems.humongous_bytes, total_bytes.max(1))
            ));
        }

        // Duplicate primitive arrays
        if problems.dup_prim.wasted_bytes > 0 {
            let pct = calc_percent(problems.dup_prim.wasted_bytes, total_bytes.max(1));
            if pct >= 1.0 {
                summary.push_str(&format!(
                    "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Duplicate primitive arrays {}</font></td></tr>\n",
                    bold("Problem"),
                    severity_low(&format!("{}", format_with_percent(problems.dup_prim.wasted_bytes, total_bytes.max(1))))
                ));
                summary.push_str(&format!(
                    "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                    <tr><td class=\"TAL\"><font class=\"Bigger\">{} groups of identical primitive arrays. \
                    <button onclick=\"openSection('Section-14');\">Section 14. Duplicate Primitive Arrays</button></font></td></tr>\
                    </table></div></details></font></td></tr>\n",
                    problems.dup_prim.duplicate_group_count
                ));
            }
        }

        // Duplicate object arrays
        if problems.dup_obj.wasted_bytes > 0 {
            let pct = calc_percent(problems.dup_obj.wasted_bytes, total_bytes.max(1));
            if pct >= 0.5 {
                summary.push_str(&format!(
                    "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Duplicate object arrays {}</font></td></tr>\n",
                    bold("Problem"),
                    severity_low(&format!("{}", format_with_percent(problems.dup_obj.wasted_bytes, total_bytes.max(1))))
                ));
                summary.push_str(&format!(
                    "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                    <tr><td class=\"TAL\"><font class=\"Bigger\">{} groups of identical object arrays. \
                    <button onclick=\"openSection('Section-15');\">Section 15. Duplicate Object Arrays</button></font></td></tr>\
                    </table></div></details></font></td></tr>\n",
                    problems.dup_obj.duplicate_group_count
                ));
            }
        }
    }
}

/// Format class name for display: replace '/' with '.' and wrap in <t-c> tag.
fn format_class_name(name: &str) -> String {
    let display = name.replace('/', ".");
    format!("<t-c>{}</t-c>", display)
}

/// Basic HTML escaping for inline text (advice / JVM flag values).
fn html_escape_simple(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Categorized JVM flags parsed from the raw `-X`/`-XX` string. Each entry
/// is (flag, value, description). Known flags get a human-readable
/// description; unknown flags land in `other_flags` without one.
struct ParsedJvmFlags {
    heap_flags: Vec<(String, String, String)>,
    gc_flags: Vec<(String, String, String)>,
    perf_flags: Vec<(String, String, String)>,
    other_flags: Vec<(String, String)>,
}

fn parse_jvm_flags(raw: &str) -> ParsedJvmFlags {
    let mut heap = Vec::new();
    let mut gc = Vec::new();
    let mut perf = Vec::new();
    let mut other = Vec::new();

    for tok in raw.split_whitespace() {
        // Decompose: -XX:+Flag / -XX:-Flag / -XX:Flag=value / -Xmx<size> / -X<size>
        let (flag, value) = if let Some(rest) = tok.strip_prefix("-XX:") {
            if let Some(name) = rest.strip_prefix("+") {
                (name.to_string(), "enabled".to_string())
            } else if let Some(name) = rest.strip_prefix("-") {
                (name.to_string(), "disabled".to_string())
            } else if let Some(eq) = rest.find('=') {
                (rest[..eq].to_string(), rest[eq+1..].to_string())
            } else {
                (rest.to_string(), String::new())
            }
        } else if let Some(rest) = tok.strip_prefix("-X") {
            // -Xmx512m, -Xms128m, -Xss, -Xmn, -Xshare:on, etc.
            // Try to split into name + value
            if let Some(eq) = rest.find(|c: char| c == ':' || c == '=') {
                (format!("-X{}", &rest[..eq]), rest[eq..].to_string())
            } else if rest.starts_with("mx") || rest.starts_with("ms")
                || rest.starts_with("mn") || rest.starts_with("ss") {
                let name = &rest[..2];
                let val = &rest[2..];
                (format!("-X{}", name), val.to_string())
            } else {
                (format!("-X{}", rest), String::new())
            }
        } else {
            // Non-standard flag (e.g. just "Flag")
            (tok.to_string(), String::new())
        };

        // Categorize and provide descriptions for well-known flags.
        let upper = flag.to_uppercase();
        let (category, description) = categorize_flag(&upper);
        match category {
            FlagCategory::Heap => heap.push((flag.clone(), value.clone(), description.into())),
            FlagCategory::Gc => gc.push((flag.clone(), value.clone(), description.into())),
            FlagCategory::Perf => perf.push((flag.clone(), value.clone(), description.into())),
            FlagCategory::Other => other.push((flag.clone(), value.clone())),
        }
    }

    ParsedJvmFlags {
        heap_flags: heap,
        gc_flags: gc,
        perf_flags: perf,
        other_flags: other,
    }
}

enum FlagCategory { Heap, Gc, Perf, Other }

fn categorize_flag(upper: &str) -> (FlagCategory, &'static str) {
    // Heap-related
    if upper.starts_with("-XM") || upper.starts_with("-XX:M") || upper.starts_with("-XX:+USEMEM")
        || upper.contains("HEAPSIZE") || upper.contains("HEAPREGION")
        || upper.contains("HEAP") || upper.contains("METASPACE")
        || upper.contains("COMPRESSEDCLASSSPACE") || upper.contains("MAXDIRECTMEMORYSIZE")
        || upper.contains("OOM") || upper.contains("SOFTMX") || upper.contains("HARDMX") {
        return (FlagCategory::Heap, "Heap or stack memory sizing");
    }
    // GC
    if upper.contains("GC") || upper.contains("USEG1") || upper.contains("USEZGC")
        || upper.contains("USESHENANDOAH") || upper.contains("USEPARALLEL")
        || upper.contains("USECMS") || upper.contains("USESERNIAL")
        || upper.contains("STRINGDEDUPLICATION") || upper.contains("MAXGCPAUSEMILLIS")
        || upper.contains("G1HEAPREGIONSIZE") || upper.contains("G1RESERVEPERCENT")
        || upper.contains("CONCURRENCY") || upper.contains("PARALLELGC")
        || upper.contains("USESHARED") || upper.contains("REQUIRESHARED")
        || upper.contains("DYNAMICDUMP") || upper.contains("DUMPSHARED")
        || upper.contains("SHARE") || upper.contains("ARCHIVE")
        || upper.contains("CDS") || upper.contains("CLASSLIST") {
        return (FlagCategory::Gc, "Garbage collector / class data sharing");
    }
    // Performance / JIT
    if upper.contains("JIT") || upper.contains("TIEREDSTOPATLEVEL")
        || upper.contains("TIEREDCOMPILATION") || upper.contains("CI")
        || upper.contains("COMPILETHRESHOLD") || upper.contains("USEBIASEDLOCKING")
        || upper.contains("LOCKING") || upper.contains("USECOUNTEDLOOPSAFETY")
        || upper.contains("AGGRESSIVEOPTS") || upper.contains("USEOPTSTRING")
        || upper.contains("ENABLEPRECONDITIONS") || upper.contains("INVOKE")
        || upper.contains("FLIGHTRECORDER") || upper.contains("JFR")
        || upper.contains("LOG") || upper.contains("XLOG") {
        return (FlagCategory::Perf, "JIT compiler / logging / runtime");
    }
    (FlagCategory::Other, "")
}

/// Render a single advice as a collapsible panel (used in the Easy-to-fix area).
fn render_advice_panel(advice: &Advice, _total_bytes: i64) -> String {
    let color_class = match advice.severity {
        AdviceSeverity::High => "OvhdColorHigh",
        AdviceSeverity::Med | AdviceSeverity::Low => "OvhdColorMed",
    };
    let overhead_suffix = match advice.overhead_bytes {
        Some(b) => format!("&nbsp;&nbsp;Overhead: <b>{}</b>.", format_with_percent(b, _total_bytes.max(1))),
        None => String::new(),
    };
    format!(
        "<details>\n\
         <summary><font class=\"{}\">&nbsp;<font class=\"Bigger\"><b>{} {}{}</b></font>&nbsp;</font></summary>\n\
         <div>\n<table class=\"dt\"><br>\n\
         <tr class=\"C_02\">\n<td class=\"TAL\"><font class=\"Bigger\"><br>\n\
         {}<br>\n<br>\n\
         For full details, go to <button onclick=\"openSection('{}');\">Section {}</button><br>\n\
         </font><br>\n</td></tr>\n</table><br>\n</div>\n\n</details>",
        color_class,
        advice.category.prefix(),
        advice.title,
        overhead_suffix,
        advice.body_html,
        advice.section_id,
        advice.section_id.trim_start_matches("Section-"),
    )
}

/// Render a single advice as a table row (used in the All problems table).
fn render_advice_table_row(advice: &Advice, total_bytes: i64) -> String {
    let sev_html = match advice.severity {
        AdviceSeverity::High => severity_high(&format!("found")),
        AdviceSeverity::Med => severity_med(&format!("found")),
        AdviceSeverity::Low => severity_low(&format!("found")),
    };
    let _ = total_bytes;
    format!(
        "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: {} {}</font></td></tr>\n",
        bold("Problem"),
        advice.title,
        sev_html,
    )
}

fn format_root_description(meta: &ss_stats::ref_chain::GcRootMeta) -> String {
    let type_names: Vec<&str> = meta.root_types.iter().map(|rt| match rt {
        ss_heap_model::root::RootType::SystemClass => "System Class",
        ss_heap_model::root::RootType::JniGlobal => "JNI Global",
        ss_heap_model::root::RootType::JniLocal => "JNI Local",
        ss_heap_model::root::RootType::JavaStackFrame => "Java Stack Frame",
        ss_heap_model::root::RootType::NativeStack => "Native Stack",
        ss_heap_model::root::RootType::ThreadBlock => "Thread Block",
        ss_heap_model::root::RootType::BusyMonitor => "Busy Monitor",
        ss_heap_model::root::RootType::ThreadObj => "Thread Object",
        ss_heap_model::root::RootType::Unknown => "Unknown",
    }).collect();
    let type_str = type_names.join(", ");

    // For Java Static display roots, class_name is already a full description
    // like "Java Static [com.foo.Bar.FIELD]" — don't wrap in <t-c>.
    if meta.class_name.starts_with("Java Static") {
        meta.class_name.clone()
    } else {
        format!("{}(<t-c>{}</t-c>)", type_str, meta.class_name)
    }
}




/// Keep the `limit` deepest chains, ties in discovery order — the order the
/// section's stable sort by depth then truncation used to keep.
fn record_long_chain(chains: &mut Vec<(u32, u32)>, depth: u32, obj_idx: u32, limit: usize) {
    if chains.len() >= limit && depth <= chains[limit - 1].0 {
        return;
    }
    let pos = chains
        .iter()
        .position(|&(d, _)| d < depth)
        .unwrap_or(chains.len());
    chains.insert(pos, (depth, obj_idx));
    if chains.len() > limit {
        chains.truncate(limit);
    }
}
/// Depth memo for the long-chain section: one root distance per object.
///
/// The section needs the distance of every object to find the chains deeper than
/// its threshold, which is 4 B per object — 480 MB on a 120 M-object dump, and the
/// largest single allocation the report makes. Above DEPTH_MEMO_RAM_OBJECTS the
/// distances live in a scratch file the kernel pages in and out, the same shape the
/// pipeline already uses for its parent log and dump-order table, so the allocator
/// stays bounded while the section computes the same distances.
enum DepthMemo {
    Ram(Vec<u32>),
    File {
        map: memmap2::MmapMut,
        path: std::path::PathBuf,
    },
}

/// Objects a depth memo may hold in RAM before it moves to a scratch file.
const DEPTH_MEMO_RAM_OBJECTS: u32 = 8 << 20;

impl DepthMemo {
    fn new(objects: u32) -> Self {
        let bytes = objects as usize * std::mem::size_of::<u32>();
        if objects <= DEPTH_MEMO_RAM_OBJECTS {
            return Self::Ram(vec![0u32; objects as usize]);
        }
        let dir = std::env::var_os("SS_MAT_TEMP_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = dir.join(format!("ss-mat-depth-{}.rec", std::process::id()));
        let opened = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .and_then(|file| {
                file.set_len(bytes as u64)?;
                // SAFETY: the map is only ever read and written through this type,
                // and the file is private to this process.
                let map = unsafe { memmap2::MmapMut::map_mut(&file)? };
                Ok(map)
            });
        match opened {
            Ok(map) => Self::File { map, path },
            Err(_) => {
                let _ = std::fs::remove_file(&path);
                Self::Ram(vec![0u32; objects as usize])
            }
        }
    }

    #[inline]
    fn get(&self, idx: u32) -> u32 {
        match self {
            Self::Ram(values) => values.get(idx as usize).copied().unwrap_or(0),
            Self::File { map, .. } => {
                let off = idx as usize * 4;
                match map.get(off..off + 4) {
                    Some(bytes) => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                    None => 0,
                }
            }
        }
    }

    #[inline]
    fn set(&mut self, idx: u32, value: u32) {
        match self {
            Self::Ram(values) => {
                if let Some(slot) = values.get_mut(idx as usize) {
                    *slot = value;
                }
            }
            Self::File { map, .. } => {
                let off = idx as usize * 4;
                if off + 4 <= map.len() {
                    map[off..off + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
}

impl Drop for DepthMemo {
    fn drop(&mut self) {
        if let Self::File { path, .. } = self {
            let _ = std::fs::remove_file(&path);
        }
    }
}
#[cfg(test)]
mod result_store_tests {
    use super::*;
    #[test]
    fn report_propagates_truncated_store_and_empty_store_overrides_legacy() {
        use ss_stats::problem_types::{BadArrayRecord, RecordStore};
        let heap=Arc::new(HeapDump { classes:Vec::new(),class_by_name:Default::default(),roots:Vec::new(),
            total_object_count:0,string_table:Default::default(),identifier_size:8,oop_size:4,alignment:8,
            header_size:12,per_class_inst_count:Vec::new(),per_class_inst_bytes:Vec::new(),stack_traces:Default::default(),
            thread_serial_map:Default::default(),thread_info:Vec::new(),timestamp_millis:0,obj_dir:None });
        let stats=Arc::new(StatsEngine::new(&[]));
        stats.bad_array_records.lock().unwrap().push(BadArrayRecord {obj_id:7,class_idx:0,kind:0,size:16});
        *stats.bad_array_store.lock().unwrap()=Some(RecordStore::new().unwrap());
        let mut reporter=MainReportGenerator::new(heap,Arc::clone(&stats),"test".into());
        let mut html=HtmlStringBuilder::new();
        reporter.section_bad_obj_arrays(&mut html).unwrap();
        let mut store=RecordStore::new().unwrap();
        store.push(BadArrayRecord {obj_id:7,class_idx:0,kind:0,size:16}).unwrap();
        // Locate this process-owned unlinked store descriptor, then truncate it
        // through procfs to exercise the real read error, not a mocked callback.
        let mut truncated=false;
        for entry in std::fs::read_dir("/proc/self/fd").unwrap() {
            let path=entry.unwrap().path();
            let Ok(target)=std::fs::read_link(&path) else {continue};
            if target.to_string_lossy().contains("ss-mat-results-") {
                if let Ok(file)=std::fs::OpenOptions::new().write(true).open(path) {
                    if file.metadata().unwrap().len()==24 {file.set_len(1).unwrap();truncated=true;}
                }
            }
        }
        assert!(truncated);
        *stats.bad_array_store.lock().unwrap()=Some(store);
        assert!(reporter.try_generate().is_err());
    }
}
