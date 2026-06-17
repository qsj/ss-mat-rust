//! Encoded-string deduplication, ported from Java's `HtmlTableFormatter`.
//!
//! Repeated class names (`<t-c>`) and long scalar values (`<t-s>`) that appear
//! inside reference-chain `<details>` blocks are replaced with tiny placeholders
//! (`|*/c<idx>|` / `|*/s<idx>|`). The full strings are emitted once into a
//! `ENCODED_STRINGS` JS array, and a client-side decoder (`rplEl`) rewrites the
//! placeholders into real tags when a `<details>` is toggled (`ontoggle`).
//!
//! Encoding rules (matching `HtmlTableFormatter.text`/`code`):
//! - Class tags always attempt encoding; if registered → `|*/c<idx>|`, else
//!   literal `<t-c>str</t-c>`.
//! - Scalar tags encode only when `str.len() >= 4`; shorter values (e.g. `0`,
//!   `true`, `null`) stay literal `<t-s>str</t-s>`.
//! - The same string table backs both `c` and `s` placeholders.
//!
//! All encoding is a no-op (returns literal tags) when `html_mode` is false,
//! matching Java's `setHtmlMode`.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Process-wide registry. There is exactly one report per process run, so a
    /// thread-local is a convenient way to make encoded-tag helpers callable
    /// from free functions without threading the registry through every call.
    static REGISTRY: RefCell<EncodedStringRegistry> =
        RefCell::new(EncodedStringRegistry::new());
}

/// Builder for a single report's encoded-string table.
pub struct EncodedStringRegistry {
    /// Ordered strings, indexed by placeholder index.
    strings: Vec<String>,
    /// str -> index (dedup). -1 marks "do not encode" (registered as a sentinel).
    index: HashMap<String, usize>,
    /// html mode flag (encoding only active when true).
    html_mode: bool,
    /// Next `sd-N` id for `<details>` blocks that opt into decoding.
    next_sd_id: usize,
}

impl EncodedStringRegistry {
    fn new() -> Self {
        Self {
            strings: Vec::with_capacity(1024),
            index: HashMap::new(),
            html_mode: false,
            next_sd_id: 0,
        }
    }

    /// Enable/disable html mode. Encoding is only active while enabled.
    pub fn set_html_mode(&mut self, on: bool) {
        self.html_mode = on;
    }

    /// Allocate the next `sd-N` id for a `<details>` block. Each encoded chain
    /// `<details>` gets one and uses `ontoggle="replEncStrs(N)"`.
    pub fn next_sd_id(&mut self) -> usize {
        let id = self.next_sd_id;
        self.next_sd_id += 1;
        id
    }

    /// Render a class-name tag. Returns `|*/c<idx>|` when the string registers,
    /// else the literal `<t-c>str</t-c>`. Class names are always ≥4 chars in
    /// practice, so encoding is attempted regardless of length (unlike scalars).
    pub fn class_tag(&mut self, s: &str) -> String {
        if self.html_mode && s.chars().count() >= 4 {
            if let Some(idx) = self.lookup_or_register(s) {
                return format!("|*/c{}|", idx);
            }
        }
        format!("<t-c>{}</t-c>", s)
    }

    /// Render a scalar-value tag. Matches Java's `code()`: only encodes when in
    /// html mode and the (escaped) value is ≥4 chars; values containing a `\`
    /// stay literal; `"`/`\n`/`\r` are escaped before encoding.
    pub fn scalar_tag(&mut self, s: &str) -> String {
        if !self.html_mode || s.chars().count() < 4 {
            return format!("<t-s>{}</t-s>", s);
        }
        // A literal backslash means we can't safely round-trip — emit literal.
        if s.contains('\\') {
            return format!("<t-s>{}</t-s>", s);
        }
        // Escape embedded quotes/newlines (matches Java code() z2 branch).
        let escaped = if s.contains('"') || s.contains('\n') || s.contains('\r') {
            let mut out = String::with_capacity(s.len() + 8);
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\n' => out.push_str("\\\\n"),
                    '\r' => out.push_str("\\\\r"),
                    c => out.push(c),
                }
            }
            out
        } else {
            s.to_string()
        };
        if let Some(idx) = self.lookup_or_register(&escaped) {
            format!("|*/s{}|", idx)
        } else {
            format!("<t-s>{}</t-s>", escaped)
        }
    }

    fn lookup_or_register(&mut self, s: &str) -> Option<usize> {
        if !self.html_mode {
            return None;
        }
        if let Some(&idx) = self.index.get(s) {
            return Some(idx);
        }
        let idx = self.strings.len();
        self.strings.push(s.to_string());
        self.index.insert(s.to_string(), idx);
        Some(idx)
    }

    /// Produce the JS array body for `ENCODED_STRINGS=[...]`, JSON-escaped.
    /// Empty when nothing was encoded.
    pub fn array_js(&self) -> String {
        if self.strings.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(self.strings.len() * 24);
        for s in &self.strings {
            out.push('"');
            // JSON-escape: escape ", \, and control chars.
            for ch in s.chars() {
                match ch {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push_str("\",\n");
        }
        // trim trailing ",\n"
        let len = out.len();
        out.truncate(len.saturating_sub(2));
        out
    }
}

// ---- thread-local convenience wrappers ----

pub fn set_html_mode(on: bool) {
    REGISTRY.with(|r| r.borrow_mut().set_html_mode(on));
}

pub fn next_sd_id() -> usize {
    REGISTRY.with(|r| r.borrow_mut().next_sd_id())
}

/// Class-name tag (text() in Java).
pub fn class_tag(s: &str) -> String {
    REGISTRY.with(|r| r.borrow_mut().class_tag(s))
}

/// Scalar-value tag (code() in Java).
pub fn scalar_tag(s: &str) -> String {
    REGISTRY.with(|r| r.borrow_mut().scalar_tag(s))
}

/// Reset the registry (call before generating a fresh report).
pub fn reset() {
    REGISTRY.with(|r| {
        let mut r = r.borrow_mut();
        r.strings.clear();
        r.index.clear();
        r.html_mode = false;
        r.next_sd_id = 0;
    });
}

/// Take the encoded-strings array JS body, then clear the registry.
pub fn take_array_js() -> String {
    REGISTRY.with(|r| {
        let mut r = r.borrow_mut();
        let js = r.array_js();
        r.strings.clear();
        r.index.clear();
        js
    })
}
