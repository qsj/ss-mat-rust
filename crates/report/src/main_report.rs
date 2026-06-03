use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use ss_heap_model::heap_dump::{FieldValue, HeapDump, LocalVarDetail};
use ss_stats::engine::StatsEngine;
use ss_stats::per_class::PerClassStats;
use ss_stats::ref_chain::{ChainStep, ClassRefChains};

use crate::html_builder::HtmlStringBuilder;
use crate::table_formatter::{
    bigger_font, bold, calc_percent, format_bytes, format_count, format_count_short,
    format_percent, format_section_header, format_with_percent, severity_high, severity_low,
    severity_med,
};

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
}

impl MainReportGenerator {
    pub fn new(heap: Arc<HeapDump>, stats: Arc<StatsEngine>, dump_file_name: String) -> Self {
        Self {
            heap,
            stats,
            section_count: 0,
            dump_file_name,
        }
    }

    fn next_section_id(&mut self) -> String {
        self.section_count += 1;
        format!("Section-{}", self.section_count)
    }

    fn section_numbered_title(&self) -> String {
        format!("{}. ", self.section_count)
    }

    /// Generate the full HTML report.
    pub fn generate(&mut self) -> String {
        let mut html = HtmlStringBuilder::new();

        // Placeholder for advisor summary (prepended later)
        html.append_raw("<!-- ADVISOR_PLACEHOLDER -->");

        // --- Section 1: Top-Level Stats ---
        self.section_top_level_stats(&mut html);

        // --- Section 2: Thread throwing OutOfMemoryError ---
        self.section_oom_thread(&mut html);

        // --- Section 3: Where Memory Goes, by Class ---
        self.section_memory_by_class(&mut html);

        // --- Section 4: Where Memory Goes, by GC Root ---
        self.section_memory_by_gc_root(&mut html);

        // --- Section 5: Live vs Garbage Objects ---
        self.section_live_vs_garbage(&mut html);

        // --- Section 6: Fixed per-object overhead ---
        self.section_fixed_overhead(&mut html);

        // --- Section 7: Objects Awaiting Finalization ---
        self.section_finalization(&mut html);

        // --- Section 8: Duplicate Strings ---
        self.section_duplicate_strings(&mut html);

        // --- Section 9: Bad Collections ---
        self.section_bad_collections(&mut html);

        // --- Section 10: Bad Object Arrays ---
        self.section_bad_obj_arrays(&mut html);

        // --- Section 11: Bad Primitive Arrays ---
        self.section_bad_prim_arrays(&mut html);

        // --- Section 12: Boxed Numbers ---
        self.section_boxed_numbers(&mut html);

        // --- Section 13: Duplicate Objects ---
        self.section_duplicate_objects(&mut html);

        // --- Section 14: Duplicate Primitive Arrays ---
        self.section_dup_prim_arrays(&mut html);

        // --- Section 15: Duplicate Object Arrays ---
        self.section_dup_obj_arrays(&mut html);

        // --- Section 16: Duplicate Lists ---
        self.section_dup_lists(&mut html);

        // --- Section 17: WeakHashMaps ---
        self.section_weak_hashmaps(&mut html);

        // --- Section 18: Off-heap DirectByteBuffer memory ---
        self.section_direct_byte_buffers(&mut html);

        // --- Section 19: Humongous objects ---
        self.section_humongous_objects(&mut html);

        // --- Section 20: Heap Size Configuration ---
        self.section_heap_config(&mut html);

        // --- Section 21: Very Long Reference Chains ---
        self.section_long_ref_chains(&mut html);

        // --- Section 22: Thread stacks ---
        self.section_thread_stacks(&mut html);

        // --- Section 23: System Properties ---
        self.section_system_properties(&mut html);

        // --- Section 24: JVM Flags ---
        self.section_jvm_flags(&mut html);

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

        doc
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

        for (&serial, st) in &self.heap.stack_traces {
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

        // Filter: only classes with non-zero shallow size, sorted descending, top 500
        // Note: total_instance_bytes already includes both instance and array shallow sizes
        // (set by BFS heap_traverser), so we don't add array_live_bytes here.
        let mut classes: Vec<_> = self
            .stats
            .per_class
            .iter()
            .filter(|pc| pc.total_instance_bytes.load(Relaxed) > 0)
            .collect();
        classes.sort_by(|a, b| {
            b.total_instance_bytes
                .load(Relaxed)
                .cmp(&a.total_instance_bytes.load(Relaxed))
        });
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
                "Retained memory",
                "How much memory is used by instances of the given class, plus all other objects that those instances reference, transitively.",
            ),
            ("Class name", ""),
        ];
        html.begin_table_with_tooltips(true, display_count, headers);

        let ref_chains = self.stats.ref_chains.lock().unwrap();

        for pc in classes.iter().take(500) {
            let inst = pc.instance_count.load(Relaxed) + pc.array_live_count.load(Relaxed) as u64;
            let shallow = pc.total_instance_bytes.load(Relaxed);
            let retained = pc.retained_size.load(Relaxed);
            // Impl-incl. size: shallow + duplicate string bytes overhead for this class
            let impl_incl = shallow + pc.duplicate_string_bytes.load(Relaxed).max(0);
            let class_name = pc.class_name.replace('/', ".");

            html.append_table_row(&[
                &format_count(inst),
                &format_with_percent(shallow, total_bytes),
                &format_with_percent(impl_incl, total_bytes),
                &format_with_percent(retained, total_bytes),
                &format!("<t-c>{}</t-c>", class_name),
            ]);

            // Render reference chains if available for this class
            if let Some(chains) = ref_chains.chains.get(&pc.class_index) {
                if !chains.expensive_fields.is_empty() || !chains.full_chains.is_empty() {
                    let chain_html = render_class_ref_chains(chains, total_bytes);
                    if !chain_html.is_empty() {
                        html.append_span_row(
                            &format!("<details><summary>Reference chains</summary>{}</details>", chain_html),
                            5,
                            false,
                        );
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
            html.begin_table(true, 0, &[
                "#instances\ngarbage", "Shallow size garbage",
                "#instances\nlive", "Shallow size live",
                "Class name",
            ]);
            let mut shown = 0;
            for (pc, garb_cnt, garb_bytes, live_cnt, live_bytes) in &class_data {
                if shown >= 50 { break; }
                if *garb_cnt == 0 && *garb_bytes == 0 { continue; }
                html.append_table_row(&[
                    &format_count(*garb_cnt as u64),
                    &format_with_percent(*garb_bytes, total_bytes.max(1)),
                    &format_count(*live_cnt as u64),
                    &format_with_percent(*live_bytes, total_bytes.max(1)),
                    &format_class_name(&pc.class_name),
                ]);
                shown += 1;
            }
            html.end_table();
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
    }

    fn section_bad_collections(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);

        // Collect collection/map class stats
        let collection_classes = [
            "java/util/HashMap", "java/util/LinkedHashMap", "java/util/Hashtable",
            "java/util/HashMap$Node", "java/util/LinkedHashMap$Entry",
            "java/util/concurrent/ConcurrentHashMap", "java/util/concurrent/ConcurrentHashMap$Node",
            "java/util/TreeMap", "java/util/ArrayList", "java/util/LinkedList",
            "java/util/Vector", "java/util/HashSet", "java/util/WeakHashMap",
            "java/util/IdentityHashMap", "java/util/EnumSet", "java/util/BitSet",
            "java/util/Collections$EmptyList", "java/util/Collections$EmptyMap", "java/util/Collections$EmptySet",
            "java/util/Collections$SingletonList", "java/util/Collections$SingletonSet",
        ];

        let mut entries: Vec<(String, u64, i64)> = Vec::new();
        let mut total_collection_bytes: i64 = 0;

        for class_name in collection_classes {
            if let Some(&idx) = self.heap.class_by_name.get(class_name) {
                if let Some(pc) = self.stats.get_per_class(idx) {
                    let count = pc.instance_count.load(Relaxed);
                    let bytes = pc.total_instance_bytes.load(Relaxed);
                    if count > 0 {
                        total_collection_bytes += bytes;
                        entries.push((class_name.replace('/', "."), count, bytes));
                    }
                }
            }
        }

        let pct = calc_percent(total_collection_bytes, total_bytes);
        let indicator = if pct >= 5.0 {
            severity_med(&format!("{} ({:.1}%)", format_bytes(total_collection_bytes), pct))
        } else if total_collection_bytes > 0 {
            severity_low(&format!("{}", format_bytes(total_collection_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Bad Collections", false, &indicator);

        if !entries.is_empty() {
            entries.sort_by(|a, b| b.2.cmp(&a.2));

            // Summary table
            html.begin_table(true, 0, &["# instances", "Shallow size", "% of heap", "Class name"]);
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
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No collection instances found", ""]);
            html.end_table();
        }
    }

    fn section_bad_obj_arrays(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        let empty_count = self.stats.empty_obj_array_count.load(Relaxed);
        let empty_bytes = self.stats.empty_obj_array_bytes.load(Relaxed);
        let single_count = self.stats.single_elem_obj_array_count.load(Relaxed);
        let single_bytes = self.stats.single_elem_obj_array_bytes.load(Relaxed);
        let all_null_count = self.stats.all_null_obj_array_count.load(Relaxed);
        let all_null_bytes = self.stats.all_null_obj_array_bytes.load(Relaxed);
        let one_nonnull_count = self.stats.one_nonnull_obj_array_count.load(Relaxed);
        let one_nonnull_bytes = self.stats.one_nonnull_obj_array_bytes.load(Relaxed);
        let sparse_count = self.stats.sparse_obj_array_count.load(Relaxed);
        let sparse_bytes = self.stats.sparse_obj_array_bytes.load(Relaxed);

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
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No bad object arrays found.", ""]);
            html.end_table();
        }
    }

    fn section_bad_prim_arrays(&mut self, html: &mut HtmlStringBuilder) {
        let total_bytes = self.stats.total_shallow_size.load(Relaxed).max(1);
        let empty_count = self.stats.empty_prim_array_count.load(Relaxed);
        let empty_bytes = self.stats.empty_prim_array_bytes.load(Relaxed);
        let trailing_count = self.stats.trailing_zero_prim_array_count.load(Relaxed);
        let trailing_bytes = self.stats.trailing_zero_prim_array_bytes.load(Relaxed);
        let total_bad = empty_count + trailing_count;
        let total_bad_bytes = empty_bytes + trailing_bytes;

        let indicator = if total_bad > 0 && calc_percent(total_bad_bytes, total_bytes) >= 1.0 {
            severity_low(&format!("Overhead {:.1}% ({})", calc_percent(total_bad_bytes, total_bytes), format_bytes(total_bad_bytes)))
        } else if total_bad > 0 {
            severity_low(&format!("{}", format_bytes(total_bad_bytes)))
        } else {
            String::new()
        };

        self.begin_section(html, "Bad Primitive Arrays", false, &indicator);

        if total_bad > 0 {
            html.begin_table(true, 0, &["Problem type", "# arrays", "Size"]);
            if empty_count > 0 {
                html.append_table_row(&[
                    "Empty primitive arrays (length=0)",
                    &format_count(empty_count),
                    &format_with_percent(empty_bytes, total_bytes),
                ]);
            }
            if trailing_count > 0 {
                html.append_table_row(&[
                    "Trailing-zero primitive arrays (>50% tail zeros)",
                    &format_count(trailing_count),
                    &format_with_percent(trailing_bytes, total_bytes),
                ]);
            }
            html.end_table();

            html.append_raw("<details><summary><b>What can I do to fix?</b></summary>\n<div>\n");
            html.append_raw("<table class=\"dt\"><tr><td class=\"TAL\"><font class=\"Bigger\">\
                Empty primitive arrays waste memory due to object headers. Use null references or shared empty arrays.<br>\
                Trailing-zero arrays (where more than 50% of elements at the end are zero) indicate that the array was \
                allocated larger than needed. Consider sizing arrays more precisely, or using data structures that \
                compress or sparse-encode the data.\
                </font></td></tr></table>\n");
            html.append_raw("</div>\n</details>\n");
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No bad primitive arrays found.", ""]);
            html.end_table();
        }
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
        } else {
            html.begin_table(true, 5, &["", ""]);
            html.append_table_row(&["No duplicate object arrays detected", ""]);
            html.end_table();
        }
    }

    fn section_dup_lists(&mut self, html: &mut HtmlStringBuilder) {
        self.begin_section(html, "Duplicate Lists", false, "");
        html.begin_table(true, 5, &["", ""]);
        html.append_table_row(&[
            "Duplicate list analysis requires container walker integration",
            "",
        ]);
        html.end_table();
    }

    fn section_weak_hashmaps(&mut self, html: &mut HtmlStringBuilder) {
        self.begin_section(
            html,
            "WeakHashMaps with hard references from values to keys",
            false,
            "",
        );

        // Count WeakHashMap instances
        let whm_count = self.heap.class_by_name.get("java/util/WeakHashMap")
            .and_then(|&idx| self.stats.get_per_class(idx))
            .map(|pc| pc.instance_count.load(Relaxed))
            .unwrap_or(0);

        if whm_count > 0 {
            html.begin_table(true, 5, &["WeakHashMap instances", "Count"]);
            html.append_table_row(&["java.util.WeakHashMap", &format_count(whm_count)]);
            html.end_table();
            html.append_raw("<br><i>Note: Detecting hard references from values to keys requires reference chain analysis.</i>");
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
            html.begin_table(true, 5, &["Total DirectByteBuffers", "Total off-heap capacity"]);
            html.append_table_row(&[
                &format_count(report.total_count as u64),
                &format_bytes(report.total_capacity),
            ]);
            html.end_table();

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
        self.begin_section(
            html,
            "Very Long (Over 1000 Elements) Reference Chains",
            false,
            "",
        );
        html.begin_table(true, 5, &["", ""]);
        html.append_table_row(&[
            "Reference chain analysis requires dominator tree computation.",
            "This feature will be added in a future version.",
        ]);
        html.end_table();
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
            for (&serial, st) in &self.heap.stack_traces {
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
                (if is_oom { 0 } else { 1 }, name.to_string())
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
                html.append_raw("<details><summary>Raw JVM flags</summary><div class=\"scroll-wrap\">");
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

    /// Generate the "MOST IMPORTANT ISSUES" advisor summary section.
    fn generate_advisor_summary(&self) -> String {
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

        let total_bytes = self.stats.total_shallow_size.load(Relaxed);

        // Collect easy-to-fix problems
        let mut easy_fix_panels = String::new();

        // DirectByteBuffer off-heap easy fix
        let dbb_report = self.stats.direct_buffers.get_report();
        if dbb_report.total_capacity > 0 {
            let pct = calc_percent(dbb_report.total_capacity, total_bytes.max(1));
            if pct >= 5.0 {
                easy_fix_panels.push_str(&format!(
                    "<details>\n<summary><font class=\"OvhdColorHigh\">&nbsp;<font class=\"Bigger\"><b>Easy-to-fix problem found: unreachable DirectByteBuffers holding off-heap memory.  Overhead: <b>{}</b>.</b></font>&nbsp;</font></summary>\n\
                    <div>\n<table class=\"dt\"><br>\n<tr class=\"C_02\">\n<td class=\"TAL\"><font class=\"Bigger\"><br>\n\
                    We found a potential problem: <t-c>DirectByteBuffer</t-c> objects, that are<br>\n\
                    unreachable (garbage), still hold native (off-heap) memory.<br>\n\
                    Total memory used by these buffers is <b>{}</b><br>\n<br>\n\
                    To reduce off-heap memory wasted by these DirectByteBuffers, you may need to<br>\n\
                    reduce the heap size for your application (specified via the -Xmx JVM flag)<br>\n\
                    so that GC happens more frequently and the problematic DirectByteBuffer<br>\n\
                    objects are GCed sooner, or you may try to reduce the number of DirectByteBuffers.<br>\n<br>\n\
                    For full details, go to <button onclick=\"openSection('Section-18');\">Section 18. Off-heap memory used by java.nio.DirectByteBuffers</button><br>\n\
                    </font><br>\n</td></tr>\n</table><br>\n</div>\n\n</details><br>\n",
                    format_with_percent(dbb_report.total_capacity, total_bytes.max(1)),
                    format_with_percent(dbb_report.total_capacity, total_bytes.max(1))
                ));
            }
        }

        // JVM flags not provided easy fix
        let info = self.stats.jvm_info.get();
        if info.jvm_flags_raw.is_empty() {
            easy_fix_panels.push_str(&format!(
                "<details>\n<summary><font class=\"OvhdColorMed\">&nbsp;<font class=\"Bigger\"><b>Preventable problem found: JVM flags are not provided</b></font>&nbsp;</font></summary>\n\
                <div>\n<table class=\"dt\"><br>\n<tr class=\"C_02\">\n<td class=\"TAL\"><font class=\"Bigger\"><br>\n\
                The JVM flags (java command line arguments) for the app that generated this heap dump<br>\n\
                are not provided.<br>\n<br>\n\
                When the JVM flags are available, we can check for some errors and do several types of analysis<br>\n\
                much more precisely. To include them, add the following code early in your app:<br>\n<br>\n\
                <code>&nbsp&nbspimport java.lang.management.ManagementFactory;<br>\n\
                &nbsp&nbspList&lt;String&gt; jvmArgs = ManagementFactory.getRuntimeMXBean().getInputArguments();<br>\n\
                &nbsp&nbspSystem.getProperties().put(\"java.vm.inputarguments\", String.join(\" \", jvmArgs));</code><br>\n<br>\n\
                For full details, go to <button onclick=\"openSection('Section-24');\">Section 24. JVM Flags</button><br>\n\
                </font><br>\n</td></tr>\n</table><br>\n</div>\n\n</details><br>\n"
            ));
        }

        if !easy_fix_panels.is_empty() {
            summary.push_str(&bigger_font(&bold("Good news! Some problems that we found are easy to fix, see below:")));
            summary.push_str("<br>\n");
            summary.push_str(&easy_fix_panels);
        }

        summary.push_str(&bigger_font(&bold("All problems:")));
        summary.push_str("<br>\n");

        // Build problem summary table
        summary.push_str("<table class=\"dt\">\n");

        let garbage_bytes = self.stats.total_garbage_bytes.load(Relaxed);
        let garbage_pct = calc_percent(garbage_bytes, total_bytes);

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
                format_with_percent(garbage_bytes, total_bytes.max(1))
            ));
        }

        // Check for humongous objects
        let humongous_bytes: i64 = self
            .stats
            .per_class
            .iter()
            .map(|pc| pc.humongous_bytes.load(Relaxed))
            .sum();
        if humongous_bytes > 0 {
            summary.push_str(&format!(
                "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Humongous objects (&gt;1MB) found {}</font></td></tr>\n",
                bold("Problem"),
                severity_low(&format!("found - {}", format_bytes(humongous_bytes)))
            ));
            summary.push_str(&format!(
                "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                <tr><td class=\"TAL\"><font class=\"Bigger\">Objects larger than 1MB occupy {} of heap. For full details, go to \
                <button onclick=\"openSection('Section-19');\">Section 19. Humongous objects</button></font></td></tr>\
                </table></div></details></font></td></tr>\n",
                format_with_percent(humongous_bytes, total_bytes.max(1))
            ));
        }

        // Check for duplicate strings
        let string_report = self.stats.string_dup.get_report();
        if string_report.wasted_bytes > 0 {
            let pct = calc_percent(string_report.wasted_bytes, total_bytes.max(1));
            if pct >= 1.0 {
                let sev = if pct >= 5.0 { severity_high(&format!("found - {}", format_bytes(string_report.wasted_bytes))) } else { severity_med(&format!("found - {}", format_bytes(string_report.wasted_bytes))) };
                summary.push_str(&format!(
                    "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Duplicate strings {}</font></td></tr>\n",
                    bold("Problem"), sev
                ));
                summary.push_str(&format!(
                    "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                    <tr><td class=\"TAL\"><font class=\"Bigger\">{} duplicate string groups wasting {}. \
                    <button onclick=\"openSection('Section-8');\">Section 8. Duplicate Strings</button></font></td></tr>\
                    </table></div></details></font></td></tr>\n",
                    string_report.duplicate_group_count,
                    format_with_percent(string_report.wasted_bytes, total_bytes.max(1))
                ));
            }
        }

        // Check for DirectByteBuffers (in problems list too)
        if dbb_report.total_capacity > 0 {
            let pct = calc_percent(dbb_report.total_capacity, total_bytes.max(1));
            if pct >= 5.0 {
                summary.push_str(&format!(
                    "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Large DirectByteBuffer off-heap usage {}</font></td></tr>\n",
                    bold("Problem"),
                    severity_high(&format!("{} off-heap", format_bytes(dbb_report.total_capacity)))
                ));
                summary.push_str(&format!(
                    "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                    <tr><td class=\"TAL\"><font class=\"Bigger\">{} DirectByteBuffer instances using {} off-heap memory. \
                    <button onclick=\"openSection('Section-18');\">Section 18. DirectByteBuffers</button></font></td></tr>\
                    </table></div></details></font></td></tr>\n",
                    dbb_report.total_count,
                    format_bytes(dbb_report.total_capacity)
                ));
            }
        }

        // Check for boxed numbers overhead
        let boxed_overhead: i64 = self.stats.per_class.iter()
            .map(|pc| pc.boxed_number_overhead.load(Relaxed))
            .sum();
        if boxed_overhead > 0 {
            let pct = calc_percent(boxed_overhead, total_bytes.max(1));
            if pct >= 1.0 {
                summary.push_str(&format!(
                    "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Boxed number overhead {}</font></td></tr>\n",
                    bold("Problem"),
                    severity_low(&format!("{}", format_with_percent(boxed_overhead, total_bytes.max(1))))
                ));
                summary.push_str(&format!(
                    "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                    <tr><td class=\"TAL\"><font class=\"Bigger\">Boxed numbers waste memory due to object headers. \
                    <button onclick=\"openSection('Section-12');\">Section 12. Boxed Numbers</button></font></td></tr>\
                    </table></div></details></font></td></tr>\n"
                ));
            }
        }

        // Check for duplicate arrays
        let dup_prim_report = self.stats.dup_prim_arrays.get_report();
        if dup_prim_report.wasted_bytes > 0 {
            let pct = calc_percent(dup_prim_report.wasted_bytes, total_bytes.max(1));
            if pct >= 1.0 {
                summary.push_str(&format!(
                    "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Duplicate primitive arrays {}</font></td></tr>\n",
                    bold("Problem"),
                    severity_low(&format!("{}", format_with_percent(dup_prim_report.wasted_bytes, total_bytes.max(1))))
                ));
                summary.push_str(&format!(
                    "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                    <tr><td class=\"TAL\"><font class=\"Bigger\">{} groups of identical primitive arrays. \
                    <button onclick=\"openSection('Section-14');\">Section 14. Duplicate Primitive Arrays</button></font></td></tr>\
                    </table></div></details></font></td></tr>\n",
                    dup_prim_report.duplicate_group_count
                ));
            }
        }

        // Check for duplicate object arrays
        let dup_obj_report = self.stats.dup_obj_arrays.get_report();
        if dup_obj_report.wasted_bytes > 0 {
            let pct = calc_percent(dup_obj_report.wasted_bytes, total_bytes.max(1));
            if pct >= 0.5 {
                summary.push_str(&format!(
                    "<tr class=\"C_02\"><td class=\"TAL\"><font class=\"Bigger\">{}: Duplicate object arrays {}</font></td></tr>\n",
                    bold("Problem"),
                    severity_low(&format!("{}", format_with_percent(dup_obj_report.wasted_bytes, total_bytes.max(1))))
                ));
                summary.push_str(&format!(
                    "<tr class=\"C_03\"><td class=\"TAL\"><font class=\"Bigger\"><details><summary>Details</summary><div><table class=\"dt\">\
                    <tr><td class=\"TAL\"><font class=\"Bigger\">{} groups of identical object arrays. \
                    <button onclick=\"openSection('Section-15');\">Section 15. Duplicate Object Arrays</button></font></td></tr>\
                    </table></div></details></font></td></tr>\n",
                    dup_obj_report.duplicate_group_count
                ));
            }
        }

        summary.push_str("</table>\n");
        summary.push_str("</font>\n");
        summary.push_str("</details>\n");

        summary
    }
}

/// Format class name for display: replace '/' with '.' and wrap in <t-c> tag.
fn format_class_name(name: &str) -> String {
    let display = name.replace('/', ".");
    format!("<t-c>{}</t-c>", display)
}

/// Render reference chains for a single class as HTML.
fn render_class_ref_chains(chains: &ClassRefChains, total_bytes: i64) -> String {
    let mut html = String::new();

    // Part 1: Expensive data fields
    if !chains.expensive_fields.is_empty() {
        html.push_str("<details><summary>Expensive data fields</summary><br><div>");
        html.push_str(&render_chain_groups(&chains.expensive_fields, total_bytes));
        html.push_str("</div></details>");
    }

    // Part 2: Full reference chains
    if !chains.full_chains.is_empty() {
        if !html.is_empty() {
            html.push_str("<br>");
        }
        html.push_str("<details><summary>Full reference chains</summary><br><div>");
        html.push_str(&render_chain_groups(&chains.full_chains, total_bytes));
        html.push_str("</div></details>");
    }

    html
}

/// Render a list of chain groups as an HTML table.
fn render_chain_groups(groups: &[ss_stats::ref_chain::RefChainGroup], total_bytes: i64) -> String {
    let mut html = String::new();
    html.push_str("<table class=\"dt\">");

    for group in groups {
        // Description row: collapsible summary with samples
        let sample_html = render_samples(&group.samples);
        let size_str = format_bytes(group.size_bytes);
        let pct = calc_percent(group.size_bytes, total_bytes);
        let desc = format!(
            "<details><summary>{} ({:.1}%): {} objects</summary>{}</details>",
            size_str, pct, group.count, sample_html,
        );
        html.push_str("<tr class=\"C_02\"><td class=\"TAL\">");
        html.push_str(&desc);
        html.push_str("</td></tr>\n");

        // Reference path row
        let path_strs: Vec<String> = group.path.iter().map(|step| {
            match step {
                ChainStep::Field { class_name, field_name } =>
                    format!("&nwarr;<t-c>{}</t-c><t-f>.{}</t-f>", class_name, field_name),
                ChainStep::ArrayElement { class_name } =>
                    format!("&nwarr;<t-c>{}</t-c>[...]", class_name),
                ChainStep::Root { description } =>
                    format!("&nwarr;{}", description),
            }
        }).collect();

        if path_strs.is_empty() {
            continue;
        }

        if path_strs.len() == 1 {
            // Single step: render inline without collapsible
            html.push_str("<tr class=\"C_03\"><td class=\"TAL\">&nbsp;&nbsp;");
            html.push_str(&path_strs[0]);
            html.push_str("</td></tr>\n");
        } else {
            // Multi-step: render with collapsible details
            html.push_str("<tr class=\"C_03\"><td class=\"TAL\"><details><summary>&nbsp;&nbsp;");
            html.push_str(&path_strs[0]);
            html.push_str("</summary>");
            for (i, step_str) in path_strs.iter().skip(1).enumerate() {
                let tab_level = (3 + i).min(10);
                html.push_str(&format!("<rc_tab{}>", tab_level));
                html.push_str(step_str);
                html.push_str("<br></rc_tab");
                html.push_str(&tab_level.to_string());
                html.push_str(">\n");
            }
            html.push_str("</details></td></tr>\n");
        }
    }

    html.push_str("</table>");
    html
}

/// Render random sample descriptions as a sub-table.
fn render_samples(samples: &[String]) -> String {
    if samples.is_empty() {
        return String::new();
    }
    let mut html = String::from("<div><table class=\"dt\">");
    html.push_str("<tr class=\"C_01\"><th class=\"TAL\"><b>&nbsp;Random sample&nbsp;</b></th></tr>");
    for sample in samples.iter().take(3) {
        html.push_str("<tr class=\"C_04\"><td class=\"TSP_1\"><rc_tab2>");
        html.push_str(sample);
        html.push_str("</rc_tab2></td></tr>");
    }
    html.push_str("</table></div>");
    html
}

/// Render reference chains for humongous objects directly into the HtmlStringBuilder.
fn render_humongous_ref_chains(chains: &ClassRefChains, html: &mut HtmlStringBuilder, total_bytes: i64) {
    // Expensive data fields: show only the first (closest) path step, non-collapsible
    if !chains.expensive_fields.is_empty() {
        html.append_raw("<details><summary>Expensive data fields</summary><br><div>");
        html.begin_table(true, 0, &[]);
        for group in &chains.expensive_fields {
            let sample_html = render_samples(&group.samples);
            let size_str = format_bytes(group.size_bytes);
            let pct = calc_percent(group.size_bytes, total_bytes);
            let desc = format!(
                "<details><summary>{} ({:.1}%): {} objects</summary>{}</details>",
                size_str, pct, group.count, sample_html,
            );
            html.append_span_row(&desc, 1, false);

            // Expensive panel: only the first path step, rendered as a simple non-collapsible row
            if let Some(step) = group.path.first() {
                let step_str = match step {
                    ChainStep::Field { class_name, field_name } =>
                        format!("&nwarr;<t-c>{}</t-c><t-f>.{}</t-f>", class_name, field_name),
                    ChainStep::ArrayElement { class_name } =>
                        format!("&nwarr;<t-c>{}</t-c>[...]", class_name),
                    ChainStep::Root { description } =>
                        format!("&nwarr;{}", description),
                };
                // Single-step: non-collapsible row matching Java's C_03 style
                let row_class = if html.alternate_row() { "C_02" } else { "C_03" };
                html.append_raw(&format!(
                    "<tr class=\"{}\"><td class=\"TAL\">&nbsp&nbsp{}</td></tr>\n",
                    row_class, step_str
                ));
            }
        }
        html.end_table();
        html.append_raw("</div></details>");
    }

    // Full reference chains: show all path steps, collapsible
    if !chains.full_chains.is_empty() {
        html.append_raw("<br><details><summary>Full reference chains</summary><br><div>");
        html.begin_table(true, 0, &[]);
        for group in &chains.full_chains {
            let sample_html = render_samples(&group.samples);
            let size_str = format_bytes(group.size_bytes);
            let pct = calc_percent(group.size_bytes, total_bytes);
            let desc = format!(
                "<details><summary>{} ({:.1}%): {} objects</summary>{}</details>",
                size_str, pct, group.count, sample_html,
            );
            html.append_span_row(&desc, 1, false);

            // Note: append_reference_path_row adds its own &nwarr; prefix per step
            let path_strs: Vec<String> = group.path.iter().map(|step| {
                match step {
                    ChainStep::Field { class_name, field_name } =>
                        format!("<t-c>{}</t-c><t-f>.{}</t-f>", class_name, field_name),
                    ChainStep::ArrayElement { class_name } =>
                        format!("<t-c>{}</t-c>[...]", class_name),
                    ChainStep::Root { description } =>
                        description.clone(),
                }
            }).collect();

            if !path_strs.is_empty() {
                let chain_refs: Vec<String> = path_strs.iter().map(|s| s.as_str().to_string()).collect();
                html.append_reference_path_row(&chain_refs, None, path_strs.len() > 1);
            }
        }
        html.end_table();
        html.append_raw("</div></details>");
    }
}

// ============================================================
// Section 4 helpers
// ============================================================

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

    format!("{}(<t-c>{}</t-c>)", type_str, meta.class_name)
}

fn render_tree_node(node: &ss_stats::ref_chain::GcRootTreeNode, total_bytes: i64) -> String {
    let mut html = String::new();
    let retained_pct = calc_percent(node.retained_bytes, total_bytes);
    let self_pct = calc_percent(node.self_bytes, total_bytes);

    if node.children.is_empty() && node.collapsed_count == 0 {
        // Leaf node
        if let Some(ref field_name) = node.field_name {
            html.push_str(&format!(
                "<li><t-f>{}</t-f>&nbsp;&searr;{}&nbsp;({}),&nbsp;{}&nbsp;reference(s)</li>\n",
                field_name,
                format_bytes(node.retained_bytes),
                format_percent(retained_pct),
                format_count(node.self_count),
            ));
        } else {
            html.push_str(&format!(
                "<li><t-c>{}</t-c>&nbsp;&searr;{}&nbsp;({}),&nbsp;self&nbsp;{}&nbsp;({}),&nbsp;{}&nbsp;object(s)</li>\n",
                node.class_name,
                format_bytes(node.retained_bytes),
                format_percent(retained_pct),
                format_bytes(node.self_bytes),
                format_percent(self_pct),
                format_count(node.self_count),
            ));
        }
    } else {
        // Node with children — collapsible
        if let Some(ref field_name) = node.field_name {
            html.push_str(&format!(
                "<li><label><t-f>{}</t-f>&nbsp;&searr;{}&nbsp;({}),&nbsp;{}&nbsp;reference(s)</label><input type=\"checkbox\" checked><ol>\n",
                field_name,
                format_bytes(node.retained_bytes),
                format_percent(retained_pct),
                format_count(node.self_count),
            ));
        } else {
            html.push_str(&format!(
                "<li><label><t-c>{}</t-c>&nbsp;&searr;{}&nbsp;({}),&nbsp;self&nbsp;{}&nbsp;({}),&nbsp;{}&nbsp;object(s)</label><input type=\"checkbox\" checked><ol>\n",
                node.class_name,
                format_bytes(node.retained_bytes),
                format_percent(retained_pct),
                format_bytes(node.self_bytes),
                format_percent(self_pct),
                format_count(node.self_count),
            ));
        }

        for child in &node.children {
            html.push_str(&render_tree_node(child, total_bytes));
        }

        if node.collapsed_count > 0 {
            let coll_pct = calc_percent(node.collapsed_bytes, total_bytes);
            html.push_str(&format!(
                "<li>...&nbsp;{}&nbsp;more referenced objects retaining&nbsp;{}&nbsp;({})</li>\n",
                format_count(node.collapsed_count),
                format_bytes(node.collapsed_bytes),
                format_percent(coll_pct),
            ));
        }

        html.push_str("</ol></li>\n");
    }

    html
}
