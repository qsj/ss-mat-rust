use std::collections::HashMap;
use ss_heap_model::root::RootType;

/// One step in a reference chain.
#[derive(Clone, Debug)]
pub enum ChainStep {
    /// Instance field reference: class_name.field_name
    Field { class_name: String, field_name: String },
    /// Array element reference (object/primitive array)
    ArrayElement { class_name: String },
    /// GC root entry point
    Root { description: String },
}

/// A group of similar reference chains clustered by path pattern.
#[derive(Clone, Debug)]
pub struct RefChainGroup {
    /// Total bytes consumed by objects in this group
    pub size_bytes: i64,
    /// Number of objects in this group
    pub count: u64,
    /// Random sample descriptions
    pub samples: Vec<String>,
    /// Object addresses of the sampled objects (parallel to `samples` when
    /// available), used to render per-object reference chains with field
    /// values in Section 3. Empty when not collected.
    pub sample_obj_ids: Vec<u64>,
    /// Chain path from leaf (target class) up toward root
    pub path: Vec<ChainStep>,
}

/// Per-class reference chain data.
#[derive(Clone, Debug, Default)]
pub struct ClassRefChains {
    /// "Expensive data fields" — direct field references holding large data (short chains)
    pub expensive_fields: Vec<RefChainGroup>,
    /// "Full reference chains" — complete paths from GC root to target class (long chains)
    pub full_chains: Vec<RefChainGroup>,
}

/// Global store of reference chains, keyed by class_index.
#[derive(Default)]
pub struct RefChainStore {
    pub chains: HashMap<u32, ClassRefChains>,
    /// Reference chains specifically for humongous objects (Section 19)
    pub humongous_chains: ClassRefChains,
}

// ============================================================
// Section 4: GC Root tree data structures
// ============================================================

/// Metadata for a single GC root object.
#[derive(Clone, Debug)]
pub struct GcRootMeta {
    pub root_obj_idx: u32,
    pub obj_id: u64,
    pub class_idx: u32,
    pub class_name: String,
    pub root_types: Vec<RootType>,
    pub thread_serial: Option<u32>,
}

/// A node in the per-root reference tree.
/// Represents a group of objects reachable via the same (field, class) edge.
#[derive(Clone, Debug)]
pub struct GcRootTreeNode {
    pub field_name: Option<String>,
    pub class_name: String,
    pub self_count: u64,
    pub self_bytes: i64,
    pub retained_bytes: i64,
    pub retained_count: u64,
    pub children: Vec<GcRootTreeNode>,
    pub collapsed_count: u64,
    pub collapsed_bytes: i64,
}

/// A single GC root entry with its reference tree.
#[derive(Clone, Debug)]
pub struct GcRootEntry {
    pub meta: GcRootMeta,
    pub retained_bytes: i64,
    pub retained_count: u64,
    pub self_bytes: i64,
    pub tree: GcRootTreeNode,
}

/// Garbage (unreachable) objects info for the virtual garbage root.
#[derive(Clone, Debug, Default)]
pub struct GarbageRootInfo {
    pub count: u64,
    pub bytes: i64,
    pub top_classes: Vec<(String, u64, i64)>,
}

/// Full store of GC root data for Section 4.
#[derive(Clone, Debug, Default)]
pub struct GcRootStore {
    /// Per-root entries, sorted by descending retained_bytes.
    pub entries: Vec<GcRootEntry>,
    pub remaining_count: u64,
    pub remaining_bytes: i64,
    pub remaining_samples: Vec<GcRootMeta>,
    pub garbage: GarbageRootInfo,
}
