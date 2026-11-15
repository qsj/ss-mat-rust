use std::collections::HashMap;
use std::sync::Arc;

use crate::class_info::JavaClassInfo;
use crate::root::HeapRootReference;

// Re-export hashbrown for address_to_index
pub type AddressToIndexMap = hashbrown::HashMap<u64, u32, ahash::RandomState>;

/// A resolved stack frame with method/class/source info.
#[derive(Debug, Clone)]
pub struct StackFrameInfo {
    pub method_name: String,
    pub method_signature: String,
    pub source_file: String,
    pub class_serial: u32,
    pub class_name: String,
    pub frame_id: u64,
    pub line_number: i32,
}

/// A thread's stack trace: thread serial -> list of frames (top = first).
#[derive(Debug, Clone)]
pub struct ThreadStackTrace {
    pub thread_serial: u32,
    pub frames: Vec<StackFrameInfo>,
}

/// Information about a single thread extracted from the heap dump.
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    pub thread_serial: u32,
    pub thread_obj_addr: u64,
    pub name: String,
    pub is_daemon: bool,
    pub priority: i32,
}

/// A single field value in a local variable object dump.
#[derive(Debug, Clone)]
pub enum FieldValue {
    Reference(Option<String>),  // class_name or None (null)
    Int(i32),
    Long(i64),
    Boolean(bool),
    Byte(i8),
    Char(u16),
    Short(i16),
    Float(u32),  // raw bits
    Double(u64), // raw bits
}

/// A local variable object with its field details.
#[derive(Debug, Clone)]
pub struct LocalVarDetail {
    pub class_name: String,
    pub address: u64,
    /// For instances: list of (field_name, value) pairs
    pub fields: Vec<(String, FieldValue)>,
    /// For object arrays: element class name + element addresses
    pub array_elements: Option<(String, Vec<Option<String>>)>,
    /// For primitive arrays: display string
    pub prim_array_display: Option<String>,
}

/// Local variables for a single stack frame, keyed by frame depth.
#[derive(Debug, Clone)]
pub struct FrameLocals {
    pub frame_depth: u32,
    pub objects: Vec<LocalVarDetail>,
}

/// The central data model holding all parsed heap dump data.
/// Produced by the HPROF parser and consumed by the analysis engine.
pub struct HeapDump {
    /// All classes found in the heap dump, indexed by class_idx.
    pub classes: Vec<JavaClassInfo>,
    /// Class name to index mapping for fast lookup.
    pub class_by_name: HashMap<Arc<str>, u32>,
    /// GC root references.
    pub roots: Vec<HeapRootReference>,
    /// Total number of objects (instances + arrays).
    pub total_object_count: u64,
    /// String table: string ID -> String content.
    pub string_table: HashMap<u64, String>,
    /// Size of object references in the dump (4 or 8 bytes).
    pub identifier_size: u8,
    /// Size of an oop (ordinary object pointer) in the JVM.
    pub oop_size: u8,
    /// Object alignment in bytes (typically 8).
    pub alignment: u8,
    /// JVM object header size in bytes (typically 12 or 16).
    pub header_size: u8,
    /// Per-class instance counts collected during parsing.
    pub per_class_inst_count: Vec<u64>,
    /// Per-class total instance shallow bytes collected during parsing.
    pub per_class_inst_bytes: Vec<i64>,
    /// Every indexed object address (instances and arrays), sorted.
    ///
    /// Membership tests are a binary search here instead of a hash probe: 8 bytes
    /// per object against roughly 24 for the map, and the pipeline only ever asks
    /// whether an address is indexed.
    pub sorted_addresses: Vec<u64>,
    /// Stack traces indexed by thread serial number.
    pub stack_traces: HashMap<u32, ThreadStackTrace>,
    /// Thread serial -> thread object address mapping (from GcRootThreadObj roots).
    pub thread_serial_map: HashMap<u32, u64>,
    /// Extracted thread info: name, daemon status, priority.
    pub thread_info: Vec<ThreadInfo>,
    /// HPROF file timestamp in millis since epoch.
    pub timestamp_millis: u64,
    /// On-disk object directory (address -> file offset/class/kind/length).
    /// When present it replaces the in-memory address maps.
    pub obj_dir: Option<ss_support::obj_dir::ObjDir>,
}

impl HeapDump {
    /// True when the address was indexed by the parser (an instance or an array).
    pub fn contains_object(&self, addr: u64) -> bool {
        self.sorted_addresses.binary_search(&addr).is_ok()
    }


    /// Class index for an object address, resolved through the on-disk object
    /// directory. Class-dump objects report None, matching the address map this
    /// replaces (which only ever held instances and arrays).
    pub fn class_idx_of(&self, addr: u64) -> Option<u32> {
        let dir = self.obj_dir.as_ref()?;
        let (_, rec) = dir.lookup(addr).ok()??;
        if rec.kind == ss_support::obj_dir::KIND_CLASS_DUMP {
            return None;
        }
        Some(rec.class_idx)
    }
}
