use hashbrown::HashMap;

/// High-performance u64 -> i32 map, replacing fastutil's Long2IntOpenHashMap.
/// Used primarily for object address-to-index lookups in the HPROF parser.
pub struct LongIntMap {
    inner: HashMap<u64, i32, ahash::RandomState>,
}

impl LongIntMap {
    pub fn new() -> Self {
        Self {
            inner: HashMap::with_hasher(ahash::RandomState::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: HashMap::with_capacity_and_hasher(capacity, ahash::RandomState::new()),
        }
    }

    #[inline]
    pub fn insert(&mut self, key: u64, value: i32) {
        self.inner.insert(key, value);
    }

    #[inline]
    pub fn get(&self, key: u64) -> Option<i32> {
        self.inner.get(&key).copied()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }
}

impl Default for LongIntMap {
    fn default() -> Self {
        Self::new()
    }
}
