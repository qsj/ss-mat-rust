//! I/O-shape tests for RecordStore.
//!
//! A full sequential pass must cost one positional read per block rather than per
//! record, rewrite/read visibility of set/get/push must not change, and the cache
//! a store keeps must not depend on the record count.

use ss_stats::problem_types::{BadArrayRecord, RecordStore, StoredRecord};

use std::sync::{Mutex, OnceLock};

/// /proc counters are process wide, so these tests run one at a time.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    // Recover from poisoning so one failing test does not fail the others.
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
}

/// (read syscalls, write syscalls) for this process.
fn io_counters() -> (u64, u64) {
    let raw = std::fs::read_to_string("/proc/self/io").unwrap();
    let field = |name: &str| {
        raw.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    };
    (field("syscr: "), field("syscw: "))
}

fn rss_kb() -> u64 {
    let raw = std::fs::read_to_string("/proc/self/status").unwrap();
    raw.lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        .unwrap_or(0)
}

fn bad_array(i: u64) -> BadArrayRecord {
    BadArrayRecord { obj_id: i, class_idx: (i % 97) as u32, kind: (i % 5) as u8, size: (i as i64) * 3 - 7 }
}

/// Eight-byte record that keeps every bit of the encoded word, so any byte
/// shifted by the block buffering shows up as a value mismatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Word(u64);

impl StoredRecord for Word {
    const WIDTH: usize = 8;
    fn encode(&self, raw: &mut [u8]) {
        raw[..8].copy_from_slice(&self.0.to_le_bytes());
    }
    fn decode(raw: &[u8]) -> Self {
        Word(u64::from_le_bytes(raw[..8].try_into().unwrap()))
    }
}

fn word(i: u64) -> Word {
    Word(0xdead_beef_0000_0000 | i)
}

/// A full sequential pass costs a handful of reads per 64 KiB block instead of
/// one read per record: before the read window this was 200_000 reads. Pushes
/// stay one positional write each, which is the store's file-visibility
/// contract, so only the read side is bounded here.
#[test]
fn bulk_push_and_full_iteration_cost_per_block() {
    let _serial = serial();
    const N: u64 = 200_000;
    let mut store = RecordStore::<BadArrayRecord>::new().unwrap();
    let (reads_before, _writes_before) = io_counters();
    for i in 0..N {
        store.push(bad_array(i)).unwrap();
    }
    let mut seen = 0u64;
    for (i, record) in store.iter().enumerate() {
        let record = record.unwrap();
        assert_eq!(record.obj_id, i as u64);
        assert_eq!(record.class_idx, (i as u64 % 97) as u32);
        assert_eq!(record.kind, (i as u64 % 5) as u8);
        assert_eq!(record.size, (i as i64) * 3 - 7);
        seen += 1;
    }
    let (reads_after, _writes_after) = io_counters();
    assert_eq!(seen, N);
    // 200_000 records of 24 bytes are 4.6 MiB: under 100 blocks. The bound leaves
    // room for the counter reads themselves and the file setup.
    assert!(reads_after - reads_before < 1_000, "iteration did {} reads for {N} records", reads_after - reads_before);
}

/// Records round-trip bit-exactly across block boundaries, including the record
/// that starts a new block.
#[test]
fn block_boundaries_keep_every_byte() {
    let _serial = serial();
    // 64 KiB of eight-byte records is 8192 records, so this spans three blocks
    // and lands the tail mid-block.
    const N: u64 = 20_000;
    let mut store = RecordStore::<Word>::new().unwrap();
    for i in 0..N {
        store.push(word(i)).unwrap();
    }
    for i in [0u64, 8_191, 8_192, 8_193, 16_383, 16_384, N - 1] {
        assert_eq!(store.get(i).unwrap(), word(i), "record {i}");
    }
    for (i, record) in store.iter().enumerate() {
        assert_eq!(record.unwrap(), word(i as u64), "record {i}");
    }
}

/// set after get keeps the rewrite visible, whether the record is in the pending
/// block, in the cached read window, or only on disk; push after get keeps both.
#[test]
fn interleaved_push_get_set_keep_visibility() {
    let _serial = serial();
    const N: u64 = 20_000;
    let mut store = RecordStore::<BadArrayRecord>::new().unwrap();
    for i in 0..N {
        store.push(bad_array(i)).unwrap();
    }
    // Fill the read window at a record that is already in the file.
    let mut first = store.get(0).unwrap();
    first.size = 1_234;
    store.set(0, first).unwrap();
    assert_eq!(store.get(0).unwrap().size, 1_234);
    // Rewrite a record after the window moved on, so the read comes from the file.
    let mut tail = store.get(N - 1).unwrap();
    tail.size = -55;
    store.set(N - 1, tail).unwrap();
    assert_eq!(store.get(N - 1).unwrap().size, -55);
    // Push after reads stays visible to the next get.
    store.push(bad_array(N)).unwrap();
    assert_eq!(store.get(N).unwrap().obj_id, N);
    assert_eq!(store.len(), N + 1);
    // Move the window far away, then read the rewritten record from disk again.
    assert_eq!(store.get(5_000).unwrap().obj_id, 5_000);
    assert_eq!(store.get(0).unwrap().size, 1_234);
    // A rewrite of a record the current window covers is visible immediately.
    let mut covered = store.get(5_000).unwrap();
    covered.size = 777;
    store.set(5_000, covered).unwrap();
    assert_eq!(store.get(5_000).unwrap().size, 777);
    // Iteration keeps sample and record order across the whole store.
    let rows: Vec<BadArrayRecord> = store.iter().map(|r| r.unwrap()).collect();
    assert_eq!(rows.len() as u64, N + 1);
    assert_eq!(rows[0].size, 1_234);
    assert_eq!(rows[(N - 1) as usize].size, -55);
    assert_eq!(rows[5_000].size, 777);
    for (i, row) in rows.iter().enumerate() {
        let i = i as u64;
        if i != 0 && i != 5_000 && i != N - 1 {
            assert_eq!(row.obj_id, i, "record {i}");
        }
    }
}

/// Index checks keep their error kinds.
#[test]
fn out_of_range_index_errors_are_unchanged() {
    let _serial = serial();
    let mut store = RecordStore::<BadArrayRecord>::new().unwrap();
    assert!(store.is_empty());
    assert_eq!(store.get(0).unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    let err = store.set(0, bad_array(0)).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::Other);
    assert_eq!(err.to_string(), "result index");
    store.push(bad_array(0)).unwrap();
    assert!(!store.is_empty());
    assert_eq!(store.get(1).unwrap_err().kind(), std::io::ErrorKind::UnexpectedEof);
    assert_eq!(store.set(1, bad_array(1)).unwrap_err().to_string(), "result index");
}

/// Store memory does not grow with the record count: a million records are 24 MiB
/// of data and must cost one fixed read block.
#[test]
fn store_memory_stays_bounded_for_a_large_record_count() {
    let _serial = serial();
    const N: u64 = 1_000_000;
    let before = rss_kb();
    let mut store = RecordStore::<BadArrayRecord>::new().unwrap();
    for i in 0..N {
        store.push(bad_array(i)).unwrap();
    }
    let after = rss_kb();
    assert_eq!(store.len(), N);
    assert_eq!(store.get(N - 1).unwrap().obj_id, N - 1);
    let grown = after.saturating_sub(before);
    assert!(grown < 8 * 1024, "a {N}-record store grew RSS by {grown} KiB");
}
