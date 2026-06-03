use std::sync::Mutex;

/// JVM information extracted from the heap dump.
pub struct JvmInfo {
    pub address_size: i32,
    pub id_size: i32,
    pub gc_type: String,
    pub xmx: i64,
    pub xms: i64,
    pub jvm_flags_raw: String,
    pub system_properties: Vec<(String, String)>,
    /// G1 heap region size in bytes (0 if not applicable)
    pub g1_region_size: i64,
}

/// Thread-safe wrapper for JVM info.
pub struct JvmInfoCollector {
    inner: Mutex<JvmInfo>,
}

impl JvmInfoCollector {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(JvmInfo {
                address_size: 0,
                id_size: 0,
                gc_type: String::new(),
                xmx: 0,
                xms: 0,
                jvm_flags_raw: String::new(),
                system_properties: Vec::new(),
                g1_region_size: 0,
            }),
        }
    }

    pub fn set(&self, info: JvmInfo) {
        *self.inner.lock().unwrap() = info;
    }

    pub fn get(&self) -> JvmInfo {
        let r = self.inner.lock().unwrap();
        JvmInfo {
            address_size: r.address_size,
            id_size: r.id_size,
            gc_type: r.gc_type.clone(),
            xmx: r.xmx,
            xms: r.xms,
            jvm_flags_raw: r.jvm_flags_raw.clone(),
            system_properties: r.system_properties.clone(),
            g1_region_size: r.g1_region_size,
        }
    }
}
