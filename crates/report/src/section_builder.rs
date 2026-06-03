/// Section builder for report content organization.
/// Tracks section state including severity, overhead tracking, and problem detection.
pub struct SectionBuilder {
    pub section_count: usize,
    pub section_anchor_id: String,
    pub has_problem: bool,
    summary_rows: Vec<String>,
    detail_html: String,
    threshold_rows: Option<Vec<Vec<String>>>,
    severity_label: String,
    short_description: String,
    indicator_text: String,
    detail_link: String,
}

impl SectionBuilder {
    pub fn new() -> Self {
        Self {
            section_count: 0,
            section_anchor_id: String::new(),
            has_problem: false,
            summary_rows: Vec::new(),
            detail_html: String::new(),
            threshold_rows: None,
            severity_label: String::new(),
            short_description: String::new(),
            indicator_text: String::new(),
            detail_link: String::new(),
        }
    }

    pub fn begin_section(&mut self) {
        self.section_count += 1;
        self.section_anchor_id = format!("Section-{}", self.section_count);
        self.has_problem = false;
        self.detail_html.clear();
        self.threshold_rows = None;
    }

    pub fn mark_as_problem(&mut self, severity: &str, short_desc: &str, indicator: &str) {
        self.has_problem = true;
        self.severity_label = severity.to_string();
        self.short_description = short_desc.to_string();
        self.indicator_text = indicator.to_string();
        self.detail_link = format!(
            "For full details, go to <button onclick=\"openSection('{}');\">Section {}</button>",
            self.section_anchor_id, self.section_count
        );
    }

    pub fn add_detail(&mut self, html: &str) {
        self.detail_html.push_str(html);
        self.detail_html.push_str("<br>\n");
    }

    pub fn add_threshold_row(&mut self, size: &str, source: &str) {
        let rows = self.threshold_rows.get_or_insert_with(|| {
            vec![vec!["Total size".to_string(), "Source".to_string()]]
        });
        rows.push(vec![size.to_string(), source.to_string()]);
    }
}
