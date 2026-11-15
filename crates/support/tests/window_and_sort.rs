//! Integration tests for the bounded-memory file utilities.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use ss_support::external_sort::{self, ExternalSorter};
use ss_support::page_window::PageWindowFile;

/// Tests that measure process-wide memory take turns, so a peak belongs to one test.
static MEMORY_SERIAL: Mutex<()> = Mutex::new(());

const MIB: usize = 1 << 20;

fn serial() -> MutexGuard<'static, ()> {
    MEMORY_SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A scratch directory under the system temp dir, removed when the test ends.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let mut path = std::env::temp_dir();
        path.push(format!("ss-support-test-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).expect("create scratch directory");
        TempDir { path }
    }

    fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    fn names(&self) -> Vec<String> {
        fs::read_dir(&self.path)
            .expect("read scratch directory")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Deterministic pseudo-random bytes (SplitMix64), so runs are reproducible.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            let take = chunk.len();
            chunk.copy_from_slice(&bytes[..take]);
        }
    }
}

fn write_pattern(path: &Path, len: u64, seed: u64) {
    let mut rng = SplitMix64::new(seed);
    let mut writer = BufWriter::with_capacity(MIB, File::create(path).expect("create pattern file"));
    let mut buf = vec![0u8; MIB];
    let mut left = len;
    while left > 0 {
        let take = left.min(buf.len() as u64) as usize;
        rng.fill(&mut buf[..take]);
        writer.write_all(&buf[..take]).expect("write pattern");
        left -= take as u64;
    }
    writer.flush().expect("flush pattern");
}

fn write_records(path: &Path, rec: usize, count: u64, seed: u64) {
    assert!(rec >= 8, "records carry a u64 key");
    let mut rng = SplitMix64::new(seed);
    let mut writer = BufWriter::with_capacity(MIB, File::create(path).expect("create record file"));
    let mut record = vec![0u8; rec];
    for index in 0..count {
        let key = rng.next_u64();
        record[..8].copy_from_slice(&key.to_le_bytes());
        for (slot, chunk) in record[8..].chunks_mut(8).enumerate() {
            let tag = index ^ (slot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let bytes = tag.to_le_bytes();
            let take = chunk.len();
            chunk.copy_from_slice(&bytes[..take]);
        }
        writer.write_all(&record).expect("write record");
    }
    writer.flush().expect("flush records");
}

fn read_record<R: Read>(reader: &mut R, buf: &mut [u8]) -> bool {
    let mut got = 0usize;
    while got < buf.len() {
        match reader.read(&mut buf[got..]) {
            Ok(0) => {
                assert_eq!(got, 0, "file ends inside a record");
                return false;
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => panic!("read failed: {e}"),
        }
    }
    true
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Order-independent record checksum: summing per-record hashes detects loss,
/// duplication and mutation regardless of the order the records land in.
fn multiset_checksum(path: &Path, rec: usize) -> (u128, u64) {
    let mut reader = BufReader::with_capacity(MIB, File::open(path).expect("open checksum file"));
    let mut record = vec![0u8; rec];
    let mut sum: u128 = 0;
    let mut count = 0u64;
    while read_record(&mut reader, &mut record) {
        sum = sum.wrapping_add(fnv1a(&record) as u128);
        count += 1;
    }
    (sum, count)
}

fn key_of(record: &[u8]) -> u64 {
    u64::from_le_bytes(record[..8].try_into().expect("8 key bytes"))
}

fn proc_status_kib(field: &str) -> usize {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find(|line| line.starts_with(field))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

#[test]
fn page_window_scan_matches_the_file() {
    let _turn = serial();
    let dir = TempDir::new("page-window");
    let path = dir.join("data.bin");
    let len: u64 = (64 * MIB as u64) + 12_345;
    write_pattern(&path, len, 0x0BAD_C0FF_EE12_3456);
    assert_eq!(fs::metadata(&path).unwrap().len(), len);

    let mut windowed = PageWindowFile::open(&path, 3 * MIB + 1234).expect("map file");
    assert_eq!(windowed.len(), len);
    assert!(!windowed.is_empty());

    let chunk = MIB;
    let mut calls = 0u64;
    let mut total = 0u64;
    let mut expected_offset = 0u64;
    let mut last_len = 0usize;
    {
        let mut source = File::open(&path).unwrap();
        let mut expected = vec![0u8; MIB];
        windowed
            .scan_seq(chunk, |offset, data| {
                assert_eq!(offset, expected_offset, "chunk {calls} starts at the wrong offset");
                assert!(data.len() <= chunk, "chunk {calls} exceeds chunk_bytes");
                source
                    .read_exact(&mut expected[..data.len()])
                    .expect("read expected bytes");
                assert_eq!(data, &expected[..data.len()], "chunk {calls} differs from the file");
                expected_offset += data.len() as u64;
                total += data.len() as u64;
                last_len = data.len();
                calls += 1;
            })
            .expect("scan sequence");
    }
    assert_eq!(total, len, "chunks do not cover the whole file");
    assert_eq!(expected_offset, len);
    assert_eq!(calls as usize, (len as usize).div_ceil(chunk));
    assert_eq!(last_len as u64, len % chunk as u64, "the ragged tail is the last chunk");

    // read_at: start of file.
    let mut buf = vec![0u8; 4096];
    assert_eq!(windowed.read_at(0, &mut buf).unwrap(), 4096);
    let mut want = vec![0u8; 4096];
    File::open(&path).unwrap().read_exact_at(&mut want, 0).unwrap();
    assert_eq!(buf, want, "read_at(0) differs from the file");

    // read_at: a range spanning a chunk boundary.
    let offset = (MIB - 7) as u64;
    let mut buf = vec![0u8; 8192];
    assert_eq!(windowed.read_at(offset, &mut buf).unwrap(), 8192);
    let mut want = vec![0u8; 8192];
    File::open(&path).unwrap().read_exact_at(&mut want, offset).unwrap();
    assert_eq!(buf, want, "read_at across a chunk boundary differs from the file");

    // read_at: short only at end of file, zero at and past end of file, zero for an empty buf.
    let mut buf = vec![0u8; 4096];
    assert_eq!(windowed.read_at(len - 10, &mut buf).unwrap(), 10);
    let mut tail = vec![0u8; 10];
    File::open(&path).unwrap().read_exact_at(&mut tail, len - 10).unwrap();
    assert_eq!(&buf[..10], &tail[..], "the short read does not match the file tail");
    assert_eq!(windowed.read_at(len, &mut buf).unwrap(), 0);
    assert_eq!(windowed.read_at(len + 1_000_000, &mut buf).unwrap(), 0);
    assert_eq!(windowed.read_at(0, &mut []).unwrap(), 0);

    // A window request below the 1 MiB floor is clamped, not rejected; 64 KiB chunks
    // then force many page-release steps over the 64 MiB file.
    let mut tight = PageWindowFile::open(&path, 1).expect("map file with a tiny window");
    let mut seen = 0u64;
    let mut tight_calls = 0u64;
    tight
        .scan_seq(64 * 1024, |offset, data| {
            assert_eq!(offset, seen);
            seen += data.len() as u64;
            tight_calls += 1;
        })
        .expect("scan sequence with a tight window");
    assert_eq!(seen, len);
    assert_eq!(tight_calls as usize, (len as usize).div_ceil(64 * 1024));

    // An empty file has nothing to visit.
    let empty_path = dir.join("empty.bin");
    File::create(&empty_path).unwrap();
    let mut empty = PageWindowFile::open(&empty_path, MIB).expect("map empty file");
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    let mut visited = 0u32;
    empty.scan_seq(4096, |_, _| visited += 1).expect("scan empty file");
    assert_eq!(visited, 0, "an empty file must not visit any chunk");
    let mut scratch = [0u8; 16];
    assert_eq!(empty.read_at(0, &mut scratch).unwrap(), 0);
    assert_eq!(empty.read_at(1 << 20, &mut scratch).unwrap(), 0);

    assert!(
        PageWindowFile::open(&dir.join("absent.bin"), MIB).is_err(),
        "opening a missing file is an error, not a panic"
    );
}

#[test]
fn external_sort_keeps_every_record() {
    let _turn = serial();
    let dir = TempDir::new("external-sort");
    let input = dir.join("in.bin");
    let out = dir.join("out.bin");
    let rec = 16usize;
    let count = 200_000u64;
    let budget = MIB;
    write_records(&input, rec, count, 0x5EED_1234_ABCD_0001);

    let before = multiset_checksum(&input, rec);
    assert_eq!(before.1, count);

    let sorter = ExternalSorter::new(rec, budget);
    assert_eq!(sorter.rec_size(), rec);
    assert_eq!(sorter.mem_budget(), budget);

    external_sort::reset_peak_buffered_record_bytes();
    let written = sorter.sort_file(&input, &out, key_of).expect("sort file");
    let peak = external_sort::peak_buffered_record_bytes();
    assert_eq!(written, count, "sort_file must report every record");
    assert!(peak <= budget, "sorter held {peak} record bytes, the budget is {budget}");

    let after = multiset_checksum(&out, rec);
    assert_eq!(after.1, count, "record count changed");
    assert_eq!(before.0, after.0, "records were lost, duplicated or altered");

    // Walk the sorted file: ascending keys plus the run boundaries seen directly.
    let mut expected_runs: Vec<(u64, u64, Vec<u8>)> = Vec::new();
    let mut previous: Option<u64> = None;
    {
        let mut reader = BufReader::with_capacity(MIB, File::open(&out).unwrap());
        let mut record = vec![0u8; rec];
        while read_record(&mut reader, &mut record) {
            let key = key_of(&record);
            if let Some(prev) = previous {
                assert!(key >= prev, "keys are not ascending: {prev} then {key}");
            }
            match expected_runs.last_mut() {
                Some(run) if run.0 == key => run.1 += 1,
                _ => expected_runs.push((key, 1, record.clone())),
            }
            previous = Some(key);
        }
    }

    let mut got_runs: Vec<(u64, u64, Vec<u8>)> = Vec::new();
    let mut grouped = 0u64;
    sorter
        .scan_groups(&out, key_of, |first, run_len| {
            assert_eq!(first.len(), rec);
            got_runs.push((key_of(first), run_len, first.to_vec()));
            grouped += run_len;
        })
        .expect("scan groups");
    assert_eq!(grouped, count, "run lengths must sum to the record count");
    assert_eq!(got_runs.len(), expected_runs.len(), "run count differs");
    for (index, (got, want)) in got_runs.iter().zip(expected_runs.iter()).enumerate() {
        assert_eq!(got.0, want.0, "run {index} has the wrong key");
        assert_eq!(got.1, want.1, "run {index} has the wrong length");
        assert_eq!(got.2, want.2, "run {index} does not start at the run's first record");
    }

    let leftovers: Vec<String> = dir
        .names()
        .into_iter()
        .filter(|name| name.starts_with("out.bin."))
        .collect();
    assert!(leftovers.is_empty(), "bucket files were left behind: {leftovers:?}");
}

#[test]
fn external_sort_handles_duplicate_keys() {
    let _turn = serial();
    let dir = TempDir::new("external-sort-dups");
    let input = dir.join("in.bin");
    let out = dir.join("out.bin");
    let rec = 8usize;
    let keys = [0u64, 1u64, 0x7F00_0000_0000_0000, u64::MAX];
    {
        let mut writer = BufWriter::new(File::create(&input).unwrap());
        for index in 0..1000usize {
            writer.write_all(&keys[index % keys.len()].to_le_bytes()).unwrap();
        }
        writer.flush().unwrap();
    }

    let sorter = ExternalSorter::new(rec, MIB);
    assert_eq!(sorter.sort_file(&input, &out, key_of).unwrap(), 1000);

    let mut runs: Vec<(u64, u64)> = Vec::new();
    sorter
        .scan_groups(&out, key_of, |first, run_len| runs.push((key_of(first), run_len)))
        .unwrap();
    assert_eq!(
        runs,
        vec![(0, 250), (1, 250), (0x7F00_0000_0000_0000, 250), (u64::MAX, 250)],
        "equal keys must collapse into one run each"
    );
}

#[test]
fn external_sort_rejects_bad_shapes() {
    let _turn = serial();
    let dir = TempDir::new("external-sort-reject");
    let input = dir.join("in.bin");
    fs::write(&input, vec![0u8; 24]).unwrap();
    let out = dir.join("out.bin");

    let err = ExternalSorter::new(0, MIB)
        .sort_file(&input, &out, |_| 0)
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "rec_size 0 must be rejected");

    // 24 bytes is not a multiple of 16.
    let ragged = ExternalSorter::new(16, MIB);
    let err = ragged.sort_file(&input, &out, |_| 0).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData, "a ragged input must be rejected");
    let err = ragged.scan_groups(&input, |_| 0, |_, _| {}).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(
        dir.names().iter().all(|name| !name.starts_with("out.bin.")),
        "a rejected sort must not leave bucket files: {:?}",
        dir.names()
    );

    // An empty input sorts to an empty output without inventing groups.
    let empty_in = dir.join("empty.bin");
    fs::write(&empty_in, b"").unwrap();
    let empty_out = dir.join("empty-out.bin");
    let sorter = ExternalSorter::new(16, MIB);
    assert_eq!(sorter.sort_file(&empty_in, &empty_out, |_| 0).unwrap(), 0);
    assert_eq!(fs::metadata(&empty_out).unwrap().len(), 0);
    let mut groups = 0u32;
    sorter.scan_groups(&empty_out, |_| 0, |_, _| groups += 1).unwrap();
    assert_eq!(groups, 0);
}

#[test]
fn external_sort_memory_stays_bounded() {
    let _turn = serial();
    let dir = TempDir::new("external-sort-mem");
    let input = dir.join("in.bin");
    let out = dir.join("out.bin");
    let rec = 8usize;
    let count = 2_000_000u64;
    let budget = MIB;
    write_records(&input, rec, count, 0xFEED_FACE_0000_0001);
    let input_bytes = fs::metadata(&input).unwrap().len();
    assert_eq!(input_bytes, count * rec as u64);

    let sorter = ExternalSorter::new(rec, budget);
    external_sort::reset_peak_buffered_record_bytes();
    let rss_before = proc_status_kib("VmRSS:");
    let hwm_before = proc_status_kib("VmHWM:");
    let written = sorter.sort_file(&input, &out, key_of).expect("sort file");
    let rss_after = proc_status_kib("VmRSS:");
    let hwm_after = proc_status_kib("VmHWM:");
    let peak = external_sort::peak_buffered_record_bytes();
    println!(
        "[bounded-memory] {} MiB input, mem_budget {} KiB: peak record buffers {} KiB;          VmRSS {rss_before} -> {rss_after} KiB; VmHWM {hwm_before} -> {hwm_after} KiB",
        input_bytes / MIB as u64,
        budget / 1024,
        peak / 1024
    );

    assert_eq!(written, count);
    assert!(peak <= budget, "sorter held {peak} record bytes, the budget is {budget}");
    let growth = rss_after.saturating_sub(rss_before);
    assert!(
        growth < input_bytes as usize / 2,
        "RSS grew by {growth} KiB while sorting {input_bytes} bytes: the input was loaded wholesale"
    );

    // The 16 MiB output is bucketed, so it must still be globally sorted.
    let mut previous: Option<u64> = None;
    let mut seen = 0u64;
    let mut reader = BufReader::with_capacity(MIB, File::open(&out).unwrap());
    let mut record = vec![0u8; rec];
    while read_record(&mut reader, &mut record) {
        let key = key_of(&record);
        if let Some(prev) = previous {
            assert!(key >= prev, "keys are not ascending: {prev} then {key}");
        }
        previous = Some(key);
        seen += 1;
    }
    assert_eq!(seen, count);
}

#[test]
fn external_sort_merges_buckets_above_the_budget() {
    let _turn = serial();
    let dir = TempDir::new("external-sort-merge");
    let input = dir.join("in.bin");
    let out = dir.join("out.bin");
    let rec = 16usize;
    let count = 50_000u64;
    let budget = 16 * 1024;
    {
        // Every key's high byte is 0, so one bucket holds the whole input: it cannot
        // fit the budget and must be sorted as runs and merged, twice over.
        let mut rng = SplitMix64::new(0xABCD_0000_1234_5678);
        let mut writer = BufWriter::new(File::create(&input).unwrap());
        for index in 0..count {
            let mut record = vec![0u8; rec];
            record[..8].copy_from_slice(&(index / 2).to_le_bytes());
            record[8..].copy_from_slice(&rng.next_u64().to_le_bytes());
            writer.write_all(&record).unwrap();
        }
        writer.flush().unwrap();
    }
    let input_bytes = fs::metadata(&input).unwrap().len();
    assert!(
        input_bytes > 20 * budget as u64,
        "the single bucket must be far larger than the budget, got {input_bytes} bytes"
    );

    let before = multiset_checksum(&input, rec);
    assert_eq!(before.1, count);

    let sorter = ExternalSorter::new(rec, budget);
    external_sort::reset_peak_buffered_record_bytes();
    let written = sorter.sort_file(&input, &out, key_of).expect("sort oversized bucket");
    let peak = external_sort::peak_buffered_record_bytes();
    assert_eq!(written, count);
    assert!(peak <= budget, "sorter held {peak} record bytes, the budget is {budget}");
    assert_eq!(multiset_checksum(&out, rec), before, "records were lost or altered");

    let mut runs: Vec<(u64, u64)> = Vec::new();
    sorter
        .scan_groups(&out, key_of, |first, run_len| runs.push((key_of(first), run_len)))
        .expect("scan groups");
    assert_eq!(runs.len() as u64, count / 2, "every key appears exactly twice");
    for (index, (key, run_len)) in runs.iter().enumerate() {
        assert_eq!(*key, index as u64, "run {index} has the wrong key");
        assert_eq!(*run_len, 2, "run {index} has the wrong length");
    }

    let leftovers: Vec<String> = dir
        .names()
        .into_iter()
        .filter(|name| name.starts_with("out.bin."))
        .collect();
    assert!(leftovers.is_empty(), "scratch files were left behind: {leftovers:?}");
}
