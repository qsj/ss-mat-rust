/// Object reference reporter for rendering reference chain trees in HTML.
pub struct ObjectReferenceReporter;

impl ObjectReferenceReporter {
    pub fn new() -> Self {
        Self
    }

    pub fn render_ref_chain(&self, _chain: &[(String, u64)]) -> String {
        "<div class='ref-chain'>Reference chain rendering not yet implemented</div>".to_string()
    }
}

impl Default for ObjectReferenceReporter {
    fn default() -> Self {
        Self::new()
    }
}
