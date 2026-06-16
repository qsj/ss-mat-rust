/// GC root reference types, matching the HPROF format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RootType {
    JniGlobal,
    JniLocal,
    JavaStackFrame,
    NativeStack,
    SystemClass,
    ThreadBlock,
    BusyMonitor,
    ThreadObj,
    Unknown,
}

/// A GC root reference with its target object and optional thread/stack info.
#[derive(Debug, Clone)]
pub struct HeapRootReference {
    pub root_type: RootType,
    pub object_id: u64,
    pub thread_serial: Option<u32>,
    pub frame_index: Option<u32>,
}
