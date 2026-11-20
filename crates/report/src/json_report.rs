//! Machine-readable export of the report.
//!
//! The rendered HTML is the canonical artefact; this module derives a structured
//! view from the very same bytes, so an agent can query the analysis without
//! parsing markup or running a browser. Deriving from the rendered document
//! rather than re-reading the statistics guarantees the two can never disagree.

/// Marker that starts a report section in the rendered document.
const SECTION_MARK: &str = r#"id="Section-"#;
const SUMMARY_ID: &str = r#"id="summary-section""#;

/// Render the JSON document for an already generated HTML report.
pub fn render(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 3);
    out.push_str("{\n");
    out.push_str(&format!("  \"generator\": {},", quote("ss-mat")));
    out.push_str(&format!("\n  \"title\": {},", quote(&text_between(html, "<title>", "</title>"))));
    out.push_str("\n  \"kpi\": [");
    let kpis = kpis(html);
    for (i, (label, value, note)) in kpis.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "\n    {{\"label\": {}, \"value\": {}, \"note\": {}}}",
            quote(label),
            quote(value),
            quote(note)
        ));
    }
    out.push_str("\n  ],");

    out.push_str("\n  \"advisor\": ");
    match advisor_block(html) {
        Some(block) => out.push_str(&section_json("MOST IMPORTANT ISSUES", block)),
        None => out.push_str("null"),
    }

    out.push_str(",\n  \"sections\": [");
    let sections = split_sections(html);
    for (i, (number, block)) in sections.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let title = section_title(block, *number);
        out.push_str("\n    ");
        out.push_str(&section_json(&title, block));
    }
    out.push_str("\n  ]\n}\n");
    out
}

fn section_json(title: &str, block: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("{{\"title\": {}, \"tables\": [", quote(title)));
    for (i, table) in tables(block).iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("\n      {\"id\": ");
        match &table.id {
            Some(id) => out.push_str(&quote(id)),
            None => out.push_str("null"),
        }
        out.push_str(", \"headers\": [");
        for (j, h) in table.headers.iter().enumerate() {
            if j > 0 {
                out.push_str(", ");
            }
            out.push_str(&quote(h));
        }
        out.push_str("], \"rows\": [");
        for (j, row) in table.rows.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str("\n        [");
            for (k, cell) in row.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                out.push_str(&quote(cell));
            }
            out.push(']');
        }
        out.push_str("]}");
    }
    out.push_str("]}");
    out
}

struct Table {
    id: Option<String>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

fn tables(block: &str) -> Vec<Table> {
    let mut out = Vec::new();
    let mut rest = block;
    while let Some(start) = rest.find("<table") {
        let after = &rest[start..];
        let Some(end) = table_extent(after) else { break };
        out.push(parse_table(&after[..end]));
        rest = &after[end..];
    }
    out
}

/// Offset just past the closing tag of the table at the front of the slice,
/// counting nesting so a collapsed evidence table cannot end its parent early.
fn table_extent(from: &str) -> Option<usize> {
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < from.len() {
        let open = from[i..].find("<table");
        let close = from[i..].find("</table>");
        match (open, close) {
            (Some(o), Some(c)) if o < c => {
                depth += 1;
                i += o + "<table".len();
            }
            (_, Some(c)) => {
                depth = depth.saturating_sub(1);
                i += c + "</table>".len();
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => return None,
        }
    }
    None
}

fn parse_table(table: &str) -> Table {
    let id = attribute(table, "id=").map(|v| unescape(&v).trim().to_string());
    let mut headers: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut rest = table;
    while let Some(start) = rest.find("<tr") {
        let after = &rest[start..];
        let Some(end) = after.find("</tr>") else { break };
        let row = &after[..end];
        let class = attribute(row, "class=").unwrap_or_default();
        let cells = cells(row);
        if class.contains("C_04") || cells.is_empty() {
            // A spanning detail row: its text belongs to the row above, not the dataset.
        } else if class.contains("C_01") && headers.is_empty() {
            headers = cells.iter().map(|c| clean_header(c)).collect();
        } else if class.contains("C_02") || class.contains("C_03") {
            rows.push(cells);
        } else if headers.is_empty() {
            headers = cells.iter().map(|c| clean_header(c)).collect();
        } else {
            rows.push(cells);
        }
        rest = &after[end + "</tr>".len()..];
    }
    Table { id, headers, rows }
}

fn cells(row: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = row;
    loop {
        let td = rest.find("<td");
        let th = rest.find("<th");
        let (start, close) = match (td, th) {
            (Some(t), Some(h)) if t <= h => (t, "</td>"),
            (Some(_), Some(h)) => (h, "</th>"),
            (Some(t), None) => (t, "</td>"),
            (None, Some(h)) => (h, "</th>"),
            (None, None) => break,
        };
        let after = &rest[start..];
        let Some(open_end) = after.find('>') else { break };
        let body = &after[open_end + 1..];
        let Some(end) = body.find(close) else { break };
        out.push(plain(&body[..end]));
        rest = &body[end + close.len()..];
    }
    out
}

/// Navigation and table headers carry sort affordances; the export keeps the name only.
fn clean_header(cell: &str) -> String {
    let mut s = cell.to_string();
    for noise in ["\u{2191}\u{2193}", "\u{24d8}", "(i)"] {
        s = s.replace(noise, "");
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn split_sections(html: &str) -> Vec<(u32, &str)> {
    let mut marks: Vec<(u32, usize)> = Vec::new();
    let mut from = 0;
    while let Some(pos) = html[from..].find(SECTION_MARK) {
        let absolute = from + pos;
        let digits_at = absolute + SECTION_MARK.len();
        let digits: String = html[digits_at..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(number) = digits.parse::<u32>() {
            marks.push((number, absolute));
        }
        from = absolute + SECTION_MARK.len();
    }
    let mut out = Vec::new();
    for (i, (number, start)) in marks.iter().enumerate() {
        let end = marks.get(i + 1).map(|(_, p)| *p).unwrap_or(html.len());
        out.push((*number, &html[*start..end]));
    }
    out
}

fn advisor_block(html: &str) -> Option<&str> {
    let start = html.find(SUMMARY_ID)?;
    let end = html[start..]
        .find(SECTION_MARK)
        .map(|p| start + p)
        .unwrap_or(html.len());
    Some(&html[start..end])
}

fn section_title(block: &str, number: u32) -> String {
    let raw = text_between(block, "<summary>", "</summary>");
    let text = plain(&raw);
    let prefix = format!("{}.", number);
    let trimmed = text.trim();
    match trimmed.strip_prefix(&prefix) {
        Some(rest) => rest.trim().to_string(),
        None => trimmed.to_string(),
    }
}

fn kpis(html: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(pos) = rest.find(r#"class="kpi-k""#) {
        let after = &rest[pos..];
        let label = plain(&text_between(after, ">", "</span>"));
        let Some(value_pos) = after.find(r#"class="kpi-v""#) else { break };
        let tail = &after[value_pos..];
        let value_block = match tail.find(r#"class="kpi-k""#) {
            Some(next) => &tail[..next],
            None => tail,
        };
        let raw = text_between(value_block, ">", "</span>");
        let note = plain(&text_between(value_block, "<em>", "</em>"));
        let mut value = plain(&raw);
        if !note.is_empty() {
            value = value.replace(&note, "").split_whitespace().collect::<Vec<_>>().join(" ");
        }
        out.push((label, value, note));
        rest = &after[value_pos..];
    }
    out
}

fn text_between(haystack: &str, open: &str, close: &str) -> String {
    let Some(start) = haystack.find(open) else { return String::new() };
    let body = &haystack[start + open.len()..];
    match body.find(close) {
        Some(end) => body[..end].to_string(),
        None => String::new(),
    }
}

fn attribute(tag: &str, name: &str) -> Option<String> {
    let pos = tag.find(name)?;
    let rest = &tag[pos + name.len()..];
    let rest = rest.trim_start();
    let quote_char = rest.chars().next()?;
    if quote_char != '"' && quote_char != '\'' {
        return None;
    }
    let body = &rest[1..];
    let end = body.find(quote_char)?;
    Some(body[..end].to_string())
}

fn plain(fragment: &str) -> String {
    let mut out = String::with_capacity(fragment.len());
    let mut in_tag = false;
    for ch in fragment.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            }
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    // Tag removal leaves a space before punctuation: "Problem : x".
    unescape(&out)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" :", ":")
        .replace(" ,", ",")
        .replace(" .", ".")
}

fn unescape(s: &str) -> String {
    let named = s
        .replace("&nbsp;", " ")
        .replace("&nbsp", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    // Numeric references carry the sort arrows and the tooltip glyph in headings.
    let mut out = String::with_capacity(named.len());
    let mut rest = named.as_str();
    while let Some(pos) = rest.find("&#") {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos + 2..];
        let Some(semi) = tail.find(';') else {
            out.push_str("&#");
            rest = tail;
            continue;
        };
        let body = &tail[..semi];
        let value = if let Some(hex) = body.strip_prefix('x').or_else(|| body.strip_prefix('X')) {
            u32::from_str_radix(hex, 16).ok()
        } else {
            body.parse::<u32>().ok()
        };
        match value.and_then(char::from_u32) {
            Some(ch) => out.push(ch),
            None => {
                out.push_str("&#");
                out.push_str(body);
                out.push(';');
            }
        }
        rest = &tail[semi + 1..];
    }
    out.push_str(rest);
    out
}

fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str(r#"\""#),
            '\\' => out.push_str(r#"\\"#),
            '\n' => out.push_str(r#"\n"#),
            '\r' => out.push_str(r#"\r"#),
            '\t' => out.push_str(r#"\t"#),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"<html><head><title>SS-Mat report for x.hprof</title></head><body>
<div class="kpis"><div class="kpi"><span class="kpi-k">Heap total</span><span class="kpi-v">1,024 KB</span></div>
<div class="kpi"><span class="kpi-k">Garbage</span><span class="kpi-v">256 KB <em>25.0%</em></span></div></div>
<details open id="summary-section"><summary>MOST IMPORTANT ISSUES</summary>
<table class="dt"><tr class="C_02"><td class="TAL">Problem: duplicate strings <font>found</font></td></tr></table>
</details>
<details id="Section-1"><summary>1. Top-Level Stats</summary>
<table class="dt" id="Table-0"><tr class="C_01"><td># instances</td><td>Shallow size</td></tr>
<tr class="C_02"><td>10</td><td>1,024KB</td></tr><tr class="C_04"><td>per-instance chains</td></tr>
<tr class="C_03"><td>20</td><td>2,048KB</td></tr></table>
</details>
</body></html>"#;

    #[test]
    fn exports_kpi_advisor_and_sections() {
        let json = render(FIXTURE);
        assert!(json.contains(r#""title": "SS-Mat report for x.hprof""#), "{json}");
        assert!(json.contains(r#""label": "Heap total", "value": "1,024 KB""#), "{json}");
        assert!(json.contains(r#""label": "Garbage", "value": "256 KB", "note": "25.0%""#), "{json}");
        assert!(json.contains(r#""title": "MOST IMPORTANT ISSUES""#), "{json}");
        assert!(json.contains(r#""title": "Top-Level Stats""#), "{json}");
        assert!(json.contains(r##""headers": ["# instances", "Shallow size"]"##), "{json}");
        // Detail rows are folded away, data rows survive in order.
        assert!(json.contains(r#"["10", "1,024KB"]"#), "{json}");
        assert!(json.contains(r#"["20", "2,048KB"]"#), "{json}");
        assert!(!json.contains("per-instance chains"), "{json}");
    }

    #[test]
    fn escapes_quotes_and_control_characters() {
        let json = quote("a\"b\\c");
        assert_eq!(json, r#""a\"b\\c""#);
    }
}
