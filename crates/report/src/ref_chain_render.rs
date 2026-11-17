//! Unified reference chain rendering for the HTML report.
//!
//! Two modes:
//!  - `render_tree` for Section 4: walks `GcRootTreeNode` recursively, output `<ol>/<li>` checkbox tree.
//!  - `render_problem_list` for problem sections (7/8/9/.../21): walks `parent[]` via
//!    `trace_chain`, output `<details><summary>...</summary><rc_tabN>...</details>` per object.
//!
//! Both modes share the `<t-c>/<t-f>/<t-s>/<rc_tabN>` tag conventions and the
//! formatting helpers in `table_formatter`.

use std::collections::HashSet;

use ss_heap_model::heap_dump::{FieldValue, HeapDump};
use ss_stats::ref_chain::{ChainStep, ClassRefChains, GcRootTreeNode};
use ss_stats::traversal_data::{FieldPool, TraversalData};
use ss_traversal::ResolvedValues;

use crate::html_builder::HtmlStringBuilder;
use crate::table_formatter::{
    calc_percent, format_bytes, format_count, format_percent,
};

/// Format a field value as an HTML string matching Java report style.
/// Object references become `<t-c>ClassName</t-c>`; every primitive (and
/// `null`) becomes a `<t-s>...</t-s>` scalar tag.
pub(crate) fn format_field_value(val: &FieldValue) -> String {
    format_field_value_simple(val)
}

/// Render a decoded object's body as `(<t-f>field</t-f> : value, ...)` for
/// instances, `[N]{...}` for arrays — matching Java's
/// `DuplicateObjectFinder.formatHeapObject`. String references whose content
/// is known become `<t-s>"text"</t-s>`; other references stay `<t-c>Class</t-c>`.
pub(crate) fn render_decoded_object_body(
    obj: &ss_traversal::DecodedObject,
    obj_addr: u64,
    rv: &ResolvedValues,
) -> String {
    // Primitive array.
    if let Some(display) = &obj.prim_array_display {
        return format!("{{{}}}", html_escape_body(display));
    }
    // Object array.
    if let Some((elem_class, elements)) = &obj.array_elements {
        let mut s = String::with_capacity(256);
        s.push_str(&format!("[{}]{{", elements.len()));
        for (i, elem) in elements.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            match elem {
                Some(name) => s.push_str(&format!("<t-c>{}</t-c>", name)),
                None => s.push_str("<t-s>null</t-s>"),
            }
        }
        s.push('}');
        return s;
    }
    // Instance with fields.
    if obj.fields.is_empty() {
        return String::new();
    }
    let mut s = String::with_capacity(512);
    s.push('(');
    for (i, (name, val)) in obj.fields.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("<t-f>{}</t-f> : ", name));
        s.push_str(&render_field_value_with_strings(val, rv));
    }
    s.push(')');
    s
}

/// Like [`format_field_value`], but when a reference points at a `String`
/// whose text is known, render `<t-s>"..."</t-s>` instead of `<t-c>String</t-c>`.
fn render_field_value_with_strings(val: &FieldValue, rv: &ResolvedValues) -> String {
    match val {
        FieldValue::Reference(None) => "<t-s>null</t-s>".to_string(),
        FieldValue::Reference(Some(class_name)) => {
            // No <t-c> tag for reference fields inside object bodies
            // (matches Java's formatHeapObject line 173/181).
            class_name.clone()
        }
        other => format_field_value_simple(other),
    }
}

/// Scalar-only formatting (no String resolution). Used where the caller has
/// already decided the value is a primitive.
///
/// Per Java's `formatHeapObject` (DuplicateObjectFinder.java line 173/181),
/// object-reference fields render their value's class name WITHOUT a `<t-c>`
/// wrapper — only primitive values and `null` get tags. This keeps `<t_c>`
/// counts close to Java's.
fn format_field_value_simple(val: &FieldValue) -> String {
    match val {
        FieldValue::Reference(None) => "<t-s>null</t-s>".to_string(),
        // Plain class display name, no <t-c> tag (matches Java line 173).
        FieldValue::Reference(Some(c)) => c.clone(),
        FieldValue::Boolean(v) => crate::encoded_strings::scalar_tag(&v.to_string()),
        FieldValue::Byte(v) => crate::encoded_strings::scalar_tag(&v.to_string()),
        FieldValue::Char(v) => {
            let ch = char::from_u32(*v as u32).unwrap_or('?');
            if *v >= 0x20 && *v < 0x7f {
                crate::encoded_strings::scalar_tag(&format!("'{}'", ch))
            } else {
                crate::encoded_strings::scalar_tag(&format!("\\u{:04x}", v))
            }
        }
        FieldValue::Short(v) => crate::encoded_strings::scalar_tag(&v.to_string()),
        FieldValue::Int(v) => crate::encoded_strings::scalar_tag(&v.to_string()),
        FieldValue::Long(v) => crate::encoded_strings::scalar_tag(&v.to_string()),
        FieldValue::Float(bits) => crate::encoded_strings::scalar_tag(&f32::from_bits(*bits).to_string()),
        FieldValue::Double(bits) => crate::encoded_strings::scalar_tag(&f64::from_bits(*bits).to_string()),
    }
}

fn html_escape_body(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

// ============================================================
// Public types
// ============================================================

/// A single problem object to render a chain for.
#[derive(Clone)]
pub struct ProblemObj {
    pub obj_idx: u32,
    pub shallow_bytes: i64,
    /// Pre-rendered description, e.g. `<t-c>byte[]</t-c>[16384]{...}`.
    pub display: String,
}

/// Input bundle for `RefChainRenderer::render_problem_list`.
pub struct RefChainRenderInput<'a> {
    /// Section title shown above the chain list, e.g. "Section 8 reference chains".
    pub title: String,
    /// Total bytes attributable to this problem set (for the summary row).
    pub size_bytes: i64,
    /// Total heap bytes (denominator for percentages).
    pub total_bytes: i64,
    /// Top-N problem objects (caller already sorted/truncated).
    pub problem_objs: &'a [ProblemObj],
    /// Cap on objects rendered (typically 20).
    pub max_objs_to_render: usize,
    /// When set, emits a `<font class="SecHdr">` sub-section header with this
    /// text and wraps the chains in a two-tier fold (mirroring Java's
    /// "Reference Chains for X" → Full reference chains structure).
    /// Example: `"Reference Chains for Duplicate Strings"`.
    pub sec_sub_title: Option<String>,
}

/// Unified reference chain renderer.
///
/// Borrows BFS data (objects/parent/parent_edge/field_pool) plus the heap to resolve
/// class names. `gc_root_set` is consulted to skip objects already rendered in the
/// Section 4 root tree (avoids duplicate chain output).
///
/// `resolved_values` (optional) carries per-object decoded field VALUES, used to
/// emit `<t-s>` scalar tags alongside each chain hop. When `None`, chains render
/// class+field-name paths only (the original behavior).
pub struct RefChainRenderer<'a> {
    pub heap: &'a HeapDump,
    /// Post-traversal data, queried on demand for object metadata and parents.
    pub data: &'a TraversalData,
    pub field_pool: &'a FieldPool,
    pub gc_root_set: &'a HashSet<u32>,
    pub resolved_values: Option<&'a ResolvedValues>,
}

impl<'a> RefChainRenderer<'a> {
    pub fn new(
        heap: &'a HeapDump,
        data: &'a TraversalData,
        field_pool: &'a FieldPool,
        gc_root_set: &'a HashSet<u32>,
    ) -> Self {
        Self {
            heap,
            data,
            field_pool,
            gc_root_set,
            resolved_values: None,
        }
    }

    /// Attach resolved per-object field values, enabling `<t-s>` scalar tags
    /// inside chains.
    pub fn with_resolved_values(mut self, rv: &'a ResolvedValues) -> Self {
        self.resolved_values = Some(rv);
        self
    }

    /// Section 4 mode: walk a `GcRootTreeNode` tree recursively.
    pub fn render_tree(&self, tree: &GcRootTreeNode, total_bytes: i64) -> String {
        render_tree_node(tree, total_bytes)
    }

    /// Problem-section mode: render a list of `<details>` blocks, one per problem object,
    /// each containing the `obj → root` chain.
    pub fn render_problem_list(&self, input: RefChainRenderInput<'_>) -> String {
        let mut html = String::with_capacity(4096);

        // Encoding is only active while rendering chain bodies (matches
        // Java's htmlMode toggle around renderProblemChainsInto).
        crate::encoded_strings::set_html_mode(true);

        // Optional sub-section header + two-tier fold wrapper, mirroring
        // Java's "Reference Chains for X" → Full reference chains structure.
        if let Some(sub) = &input.sec_sub_title {
            html.push_str(&format!(
                "<br><font class=\"SecHdr\">{}</font><br>\n",
                sub,
            ));
            // Allocate an sd id for the full-chains fold so its encoded
            // placeholders decode on toggle.
            let sd_id = crate::encoded_strings::next_sd_id();
            html.push_str(&format!(
                "<details id=\"sd-{}\" ontoggle=\"replEncStrs({})\"><summary>Full reference chains</summary><br>\n",
                sd_id, sd_id,
            ));
        }

        let total_pct = calc_percent(input.size_bytes, input.total_bytes);
        html.push_str(&format!(
            "<br><details><summary>{}&nbsp;{}&nbsp;({:.1}%)</summary><br>\n",
            input.title, format_bytes(input.size_bytes), total_pct,
        ));
        html.push_str("<div><table class=\"dt\">\n");

        let cap = input.problem_objs.len().min(input.max_objs_to_render);
        if cap == 0 {
            html.push_str("<tr class=\"C_03\"><td class=\"TAL\">&nbsp;&nbsp;No samples available.</td></tr>\n");
        }

        for (i, prob) in input.problem_objs.iter().take(cap).enumerate() {
            let row_class = if i % 2 == 0 { "C_02" } else { "C_03" };
            let size_str = format_bytes(prob.shallow_bytes);
            let pct = calc_percent(prob.shallow_bytes, input.total_bytes);
            html.push_str(&format!(
                "<tr class=\"{}\"><td class=\"TAL\"><details><summary>&nbsp;&nbsp;{}&nbsp;—&nbsp;{}&nbsp;({:.1}%)</summary>\n",
                row_class, prob.display, size_str, pct,
            ));

            // Skip if already in Section 4 root tree.
            if self.gc_root_set.contains(&prob.obj_idx) {
                html.push_str("<rc_tab3>&nbsp;&nbsp;Already shown in <button onclick=\"openSection('Section-4')\">Section 4.</button><br></rc_tab3>\n");
                html.push_str("</details></td></tr>\n");
                continue;
            }

            // Show the problem object's own decoded body (leaf of the chain)
            // so its primitive field values appear as `<t-s>` tags.
            if let Some(rv) = self.resolved_values {
                let prob_obj = self.data.object_at(prob.obj_idx);
                let prob_id = prob_obj.map(|o| o.obj_id).unwrap_or(0);
                if let Some(dec) = rv.objects.get(&prob_id) {
                    let body = render_decoded_object_body(dec, prob_id, rv);
                    if !body.is_empty() {
                        html.push_str(&format!(
                            "<rc_tab3>&nbsp;&nbsp;<t-c>{}</t-c>{}<br></rc_tab3>\n",
                            dec.class_name, body,
                        ));
                    }
                }
            }

            // Walk parent[] from problem object upward.
            let chain = self.trace_chain_from(prob.obj_idx);
            if chain.is_empty() {
                html.push_str("<rc_tab3>&nbsp;&nbsp;No reference chain available.<br></rc_tab3>\n");
            } else {
                for (depth, step) in chain.iter().enumerate() {
                    let tab = (3 + depth).min(10);
                    html.push_str(&format!("<rc_tab{}>", tab));
                    html.push_str(step);
                    html.push_str("<br></rc_tab");
                    html.push_str(&tab.to_string());
                    html.push_str(">\n");
                }
            }
            html.push_str("</details></td></tr>\n");
        }

        if input.problem_objs.len() > cap {
            let extra = input.problem_objs.len() - cap;
            html.push_str(&format!(
                "<tr class=\"C_03\"><td class=\"TAL\">&nbsp;&nbsp;... and {} more objects omitted</td></tr>\n",
                extra,
            ));
        }

        html.push_str("</table></div></details>\n");

        // Close the two-tier fold wrapper opened at the top.
        if input.sec_sub_title.is_some() {
            html.push_str("</details>\n");
        }

        // Done rendering chain bodies — disable encoding so other sections
        // emit literal tags.
        crate::encoded_strings::set_html_mode(false);
        html
    }

    /// Walk `parent[]` from `start` upward to a root. Returns one HTML-formatted
    /// string per hop. Stops at `u32::MAX` (root) or after 60 hops.
    ///
    /// When `resolved_values` is set, each hop also shows the parent object's
    /// decoded body — `(<t-f>field</t-f> : <t-s>value</t-s>, ...)` for instances
    /// — so scalar values appear as `<t-s>` tags along the chain.
    fn trace_chain_from(&self, start: u32) -> Vec<String> {
        let mut chain = Vec::new();
        let mut current = start;
        let max_hops = 60;

        for _ in 0..max_hops {
            let (p, edge_key) = self.data.parent.parent_of(current);
            if p == u32::MAX {
                break;
            }
            let parent_obj = match self.data.object_at(p) {
                Some(o) => o,
                None => break,
            };
            let class_name = get_class_display(parent_obj.class_idx, self.heap);
            let field_name = self.field_pool.get(edge_key);
            // Match Java's chain rendering: each hop is the path label plus
            // the parent object's decoded body, which carries the primitive
            // field values as `<t-s>` tags. Per Java's FieldReferenceNode +
            // formatHeapObject, the body is `(field : value, ...)` — reference
            // fields render their value's class name WITHOUT a redundant
            // `<t-c>` wrapper (matching Java line 173/181).
            let mut step = format!(
                "&nwarr;{}<t-f>.{}</t-f>",
                crate::encoded_strings::class_tag(&class_name), field_name,
            );
            if let Some(rv) = self.resolved_values {
                if let Some(dec) = rv.objects.get(&parent_obj.obj_id) {
                    let body = render_decoded_object_body(dec, parent_obj.obj_id, rv);
                    if !body.is_empty() {
                        step.push_str(&format!(
                            "&nbsp;{}{}",
                            crate::encoded_strings::class_tag(&dec.class_name),
                            body,
                        ));
                    }
                }
            }
            chain.push(step);
            current = p;
        }
        chain
    }
}

// ============================================================
// Local helpers (duplicated from heap_traverser for the report crate)
// ============================================================

fn get_class_display(class_idx: u32, heap: &HeapDump) -> String {
    if (class_idx as usize) < heap.classes.len() {
        heap.classes[class_idx as usize].name.replace('/', ".")
    } else {
        "unknown".to_string()
    }
}

// ============================================================
// Migrated free functions (used by Section 3/4/19)
// ============================================================

/// Render reference chains for a single class as HTML.
pub fn render_class_ref_chains(chains: &ClassRefChains, total_bytes: i64) -> String {
    // Encoding is active while rendering class chain paths (matches Java's
    // htmlMode around renderMemoryByClass chain rendering).
    crate::encoded_strings::set_html_mode(true);

    let mut html = String::new();

    if !chains.expensive_fields.is_empty() {
        let sd_id = crate::encoded_strings::next_sd_id();
        html.push_str(&format!(
            "<details id=\"sd-{}\" ontoggle=\"replEncStrs({})\"><summary>Expensive data fields</summary><br><div>",
            sd_id, sd_id,
        ));
        html.push_str(&render_chain_groups(&chains.expensive_fields, total_bytes));
        html.push_str("</div></details>");
    }

    if !chains.full_chains.is_empty() {
        if !html.is_empty() {
            html.push_str("<br>");
        }
        let sd_id = crate::encoded_strings::next_sd_id();
        html.push_str(&format!(
            "<details id=\"sd-{}\" ontoggle=\"replEncStrs({})\"><summary>Full reference chains</summary><br><div>",
            sd_id, sd_id,
        ));
        html.push_str(&render_chain_groups(&chains.full_chains, total_bytes));
        html.push_str("</div></details>");
    }

    crate::encoded_strings::set_html_mode(false);
    html
}

/// Render a list of chain groups as an HTML table.
pub fn render_chain_groups(groups: &[ss_stats::ref_chain::RefChainGroup], total_bytes: i64) -> String {
    let mut html = String::new();
    html.push_str("<table class=\"dt\">");

    for group in groups {
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

        let path_strs: Vec<String> = group.path.iter().map(|step| {
            match step {
                ChainStep::Field { class_name, field_name } =>
                    format!("&nwarr;{}<t-f>.{}</t-f>", crate::encoded_strings::class_tag(class_name), field_name),
                ChainStep::ArrayElement { class_name } =>
                    format!("&nwarr;{}[...]", crate::encoded_strings::class_tag(class_name)),
                ChainStep::Root { description } =>
                    format!("&nwarr;{}", description),
            }
        }).collect();

        if path_strs.is_empty() {
            continue;
        }

        if path_strs.len() == 1 {
            html.push_str("<tr class=\"C_03\"><td class=\"TAL\">&nbsp;&nbsp;");
            html.push_str(&path_strs[0]);
            html.push_str("</td></tr>\n");
        } else {
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
pub fn render_samples(samples: &[String]) -> String {
    if samples.is_empty() {
        return String::new();
    }
    let mut html = String::from("<div><table class=\"dt\">");
    html.push_str("<tr class=\"C_01\"><th class=\"TAL\"><b>&nbsp;Random sample&nbsp;</b></th></tr>");
    for sample in samples.iter().take(8) {
        html.push_str("<tr class=\"C_04\"><td class=\"TSP_1\"><rc_tab2>");
        html.push_str(sample);
        html.push_str("</rc_tab2></td></tr>");
    }
    html.push_str("</table></div>");
    html
}

/// Render reference chains for humongous objects directly into the HtmlStringBuilder.
pub fn render_humongous_ref_chains(
    chains: &ClassRefChains,
    html: &mut HtmlStringBuilder,
    total_bytes: i64,
) {
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

            if let Some(step) = group.path.first() {
                let step_str = match step {
                    ChainStep::Field { class_name, field_name } =>
                        format!("&nwarr;<t-c>{}</t-c><t-f>.{}</t-f>", class_name, field_name),
                    ChainStep::ArrayElement { class_name } =>
                        format!("&nwarr;<t-c>{}</t-c>[...]", class_name),
                    ChainStep::Root { description } =>
                        format!("&nwarr;{}", description),
                };
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

/// Recursive tree renderer for Section 4 root entries.
pub fn render_tree_node(node: &GcRootTreeNode, total_bytes: i64) -> String {
    let mut html = String::new();
    let retained_pct = calc_percent(node.retained_bytes, total_bytes);
    let self_pct = calc_percent(node.self_bytes, total_bytes);

    if node.children.is_empty() && node.collapsed_count == 0 {
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
                "<li>{}&nbsp;&searr;{}&nbsp;({}),&nbsp;self&nbsp;{}&nbsp;({}),&nbsp;{}&nbsp;object(s)</li>\n",
                crate::encoded_strings::class_tag(&node.class_name),
                format_bytes(node.retained_bytes),
                format_percent(retained_pct),
                format_bytes(node.self_bytes),
                format_percent(self_pct),
                format_count(node.self_count),
            ));
        }
    } else {
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
                "<li><label>{}&nbsp;&searr;{}&nbsp;({}),&nbsp;self&nbsp;{}&nbsp;({}),&nbsp;{}&nbsp;object(s)</label><input type=\"checkbox\" checked><ol>\n",
                crate::encoded_strings::class_tag(&node.class_name),
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
