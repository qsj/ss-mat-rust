use std::sync::atomic::{AtomicU32, Ordering};

/// Compressed object index for fast lookup and visited-bit tracking.
/// Simplified from the Java bit-packed approach — uses flat vectors with atomic status words.
pub struct ObjectIndex {
    /// Object dump metadata: (class_index, generation)
    objects: Vec<(u32, u8)>,
    /// Object status words (visited/contentProcessed flags)
    object_status: Vec<AtomicU32>,
    /// Array metadata: (class_index, generation, length)
    arrays: Vec<(u32, u8, i32)>,
    /// Array status words
    array_status: Vec<AtomicU32>,
    /// Total count
    pub total_count: u64,
}

const VISITED_BIT: u32 = 1 << 31;
const PROCESSED_BIT: u32 = 1 << 30;

impl ObjectIndex {
    pub fn new(
        objects: Vec<(u32, u8)>,
        arrays: Vec<(u32, u8, i32)>,
    ) -> Self {
        let total = objects.len() as u64 + arrays.len() as u64;
        let obj_status = (0..objects.len()).map(|_| AtomicU32::new(0)).collect();
        let arr_status = (0..arrays.len()).map(|_| AtomicU32::new(0)).collect();
        Self {
            objects,
            object_status: obj_status,
            arrays,
            array_status: arr_status,
            total_count: total,
        }
    }

    /// Decode an index: returns (class_index, is_array, generation, array_length_or_zero).
    pub fn decode(&self, index: u32) -> (u32, bool, u8, i32) {
        if index == 0 {
            return (0, false, 0, 0);
        }
        let is_array = (index & 0x80000000) != 0;
        let pos = (index & 0x7FFFFFFF) as usize - 1; // 0-based
        if is_array {
            if pos < self.arrays.len() {
                let (cls, gen, len) = self.arrays[pos];
                return (cls, true, gen, len);
            }
        } else {
            if pos < self.objects.len() {
                let (cls, gen) = self.objects[pos];
                return (cls, false, gen, 0);
            }
        }
        (0, false, 0, 0)
    }

    /// Get class index for an object.
    pub fn class_index(&self, index: u32) -> u32 {
        let (cls, _, _, _) = self.decode(index);
        cls
    }

    /// Get array length (0 if not an array).
    pub fn array_length(&self, index: u32) -> i32 {
        let (_, _, _, len) = self.decode(index);
        len
    }

    /// Atomically check and mark as visited. Returns true if was not visited.
    pub fn check_and_mark_visited(&self, index: u32) -> bool {
        let status = self.get_status_ref(index);
        if let Some(s) = status {
            // Try to set VISITED_BIT if it's not already set
            let old = s.fetch_or(VISITED_BIT, Ordering::Relaxed);
            (old & VISITED_BIT) == 0
        } else {
            false
        }
    }

    /// Check if visited.
    pub fn is_visited(&self, index: u32) -> bool {
        let status = self.get_status_ref(index);
        if let Some(s) = status {
            (s.load(Ordering::Relaxed) & VISITED_BIT) != 0
        } else {
            false
        }
    }

    /// Check if content processed.
    pub fn is_content_processed(&self, index: u32) -> bool {
        let status = self.get_status_ref(index);
        if let Some(s) = status {
            (s.load(Ordering::Relaxed) & PROCESSED_BIT) != 0
        } else {
            false
        }
    }

    /// Mark content as processed.
    pub fn mark_content_processed(&self, index: u32) {
        let status = self.get_status_ref(index);
        if let Some(s) = status {
            s.fetch_or(PROCESSED_BIT, Ordering::Relaxed);
        }
    }

    /// Get object count.
    pub fn object_count(&self) -> u64 {
        self.objects.len() as u64
    }

    /// Get array count.
    pub fn array_count(&self) -> u64 {
        self.arrays.len() as u64
    }

    fn get_status_ref(&self, index: u32) -> Option<&AtomicU32> {
        if index == 0 {
            return None;
        }
        let is_array = (index & 0x80000000) != 0;
        let pos = (index & 0x7FFFFFFF) as usize - 1;
        if is_array {
            self.array_status.get(pos)
        } else {
            self.object_status.get(pos)
        }
    }
}
