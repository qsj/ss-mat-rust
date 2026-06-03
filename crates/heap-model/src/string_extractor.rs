/// Extracts String content from heap dump objects.
/// In Java, this walks the String -> char[]/byte[] reference chain.
/// This module will be implemented in Phase 1 with the HPROF parser.

/// Extracted string content with duplication metadata.
#[derive(Debug, Clone)]
pub struct ExtractedString {
    pub value: String,
    pub object_index: u32,
    pub char_count: usize,
    pub byte_size: usize,
}
