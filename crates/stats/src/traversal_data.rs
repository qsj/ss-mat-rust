/// Per-object metadata collected during initial HPROF pass for BFS.
pub struct ObjectEntry {
    pub obj_id: u64,
    pub class_idx: u32,
    pub file_offset: usize,
    pub body_size: i32,
    /// Object type: 0=instance, 1=primitive array, 2=object array, 3=class object
    pub kind: u8,
    pub array_length: i32,
}

/// Field name pool: compact storage of field names used in edges.
/// Index 0 = "[elements]" (array elements), 1 = "[static]" (static field).
pub struct FieldPool {
    pub names: Vec<String>,
}

impl FieldPool {
    pub fn new() -> Self {
        Self {
            names: vec!["[elements]".into(), "[static]".into()],
        }
    }
    pub fn intern(&mut self, name: &str) -> u32 {
        if let Some(idx) = self.names.iter().position(|n| n == name) {
            return idx as u32;
        }
        let idx = self.names.len() as u32;
        self.names.push(name.to_string());
        idx
    }
    pub fn get(&self, key: u32) -> &str {
        &self.names[key as usize]
    }
}

impl Default for FieldPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Edge in the object graph carrying field info.
#[derive(Clone)]
pub struct OutEdge {
    pub target: u32,
    pub field_key: u32, // index into FieldPool
}

/// Persistent snapshot of BFS data exposed for post-traversal rendering
/// (e.g., reference chains in the HTML report).
///
/// Populated by `HeapObjectTraverser::run_traversal` and stored in
/// `StatsEngine::traversal_data` so the report crate can walk object→root
/// chains on demand without re-running BFS.
pub struct TraversalData {
    pub objects: Vec<ObjectEntry>,
    pub parent: Vec<u32>,
    pub parent_edge: Vec<u32>,
    pub root_of: Vec<u32>,
    pub visited: Vec<bool>,
    pub field_pool: FieldPool,
}
