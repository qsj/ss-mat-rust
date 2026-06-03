/// Format byte sizes in KB (matching Java's SizeFormatter).
pub fn format_kb(bytes: i64) -> String {
    if bytes >= 1024 {
        format!("{:.0}Kb", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Format byte sizes with KB/MB/GB units.
pub fn format_bytes(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = 1024 * KB;
    const GB: i64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0}Kb", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Format a percentage value (e.g. "38.0%").
pub fn format_percent(value: f64) -> String {
    format!("{:.1}%", value)
}

/// Format bytes with percentage of total (e.g. "67,799Kb (38.0%)").
pub fn format_with_percent(bytes: i64, total: i64) -> String {
    if total > 0 {
        let pct = bytes as f64 / total as f64 * 100.0;
        format!("{} ({:.1}%)", format_bytes(bytes), pct)
    } else {
        format_bytes(bytes)
    }
}

/// Format a count with comma separators (e.g. "2,135,318").
pub fn format_count(count: u64) -> String {
    let s = count.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

/// Format count with K/M suffixes for short display.
pub fn format_count_short(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// Calculate percentage.
pub fn calc_percent(value: i64, total: i64) -> f64 {
    if total > 0 {
        value as f64 / total as f64 * 100.0
    } else {
        0.0
    }
}

/// Wrap text in severity color class.
pub fn severity_high(text: &str) -> String {
    format!("<font class=\"OvhdColorHigh\">&nbsp;{}&nbsp;</font>", text)
}

pub fn severity_med(text: &str) -> String {
    format!("<font class=\"OvhdColorMed\">&nbsp;{}&nbsp;</font>", text)
}

pub fn severity_low(text: &str) -> String {
    format!("<font class=\"OvhdColorLow\">&nbsp;{}&nbsp;</font>", text)
}

/// Wrap in bold.
pub fn bold(text: &str) -> String {
    format!("<b>{}</b>", text)
}

/// Wrap in bigger font.
pub fn bigger_font(text: &str) -> String {
    format!("<font class=\"Bigger\">{}</font>", text)
}

/// Wrap in details/summary collapsible.
pub fn wrap_details(title: &str, content: &str) -> String {
    format!("<details><summary>{}</summary><div>{}</div></details>", title, content)
}

/// Format section header with optional documentation anchor.
pub fn format_section_header(title: &str, anchor: Option<&str>) -> String {
    if let Some(a) = anchor {
        format!("<font class=\"SecHdr\"><a href=\"{}\">{}</a></font>", a, title)
    } else {
        format!("<font class=\"SecHdr\">{}</font>", title)
    }
}
