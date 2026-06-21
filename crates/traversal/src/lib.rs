#![allow(dead_code, unused_variables, unused_assignments, unused_imports, unused_mut, unused_parens)]
pub mod engine;
pub mod linear_scan;
pub mod scan_workers;
pub mod heap_traverser;
pub mod virtual_stack;
pub mod post_aggregator;
pub mod garbage_scanner;

// Re-export the public field-value resolution API for downstream crates
// (notably `ss-report`, which needs per-object field values at render time
// to emit `<t-s>` scalar tags inside reference chains).
pub use linear_scan::{
    resolve_field_values, DecodedObject, FieldValueMap, ResolvedValues, StringContentMap,
};
