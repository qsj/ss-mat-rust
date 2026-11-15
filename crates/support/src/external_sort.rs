//! Bounded-memory external sort for fixed-size records.
//!
//! Records are routed to 256 bucket files by the high byte of their sort key, each
//! bucket is sorted in memory when it fits the budget, and the buckets are
//! concatenated in bucket order — which is globally sorted because the bucket
//! index is the key's high byte.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Routing buckets: the bucket index is the key's high byte.
const BUCKETS: usize = 256;

/// Runs merged in one pass; bounds open handles and read buffers during a merge.
const MAX_FANIN: usize = 32;

/// Hard cap on one sort buffer, so record indices stay inside a `u32`.
const MAX_RECORD_BUFFER: usize = 1 << 30;

/// Bytes of record data currently held by the sorters of this process.
static CUR_RECORD_BYTES: AtomicUsize = AtomicUsize::new(0);

/// High-water mark of `CUR_RECORD_BYTES`.
static PEAK_RECORD_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Peak bytes of record data held at once since the last reset.
///
/// Instrumentation for the memory-bound test; not part of the sorting contract.
#[doc(hidden)]
pub fn peak_buffered_record_bytes() -> usize {
    PEAK_RECORD_BYTES.load(Ordering::Relaxed)
}

/// Forget the observed peak, so the next run measures from what is held now.
#[doc(hidden)]
pub fn reset_peak_buffered_record_bytes() {
    PEAK_RECORD_BYTES.store(CUR_RECORD_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Sorts fixed-size records by a 64-bit key with bounded memory.
pub struct ExternalSorter {
    rec_size: usize,
    mem_budget: usize,
}

impl ExternalSorter {
    /// `rec_size` is the fixed record width in bytes; `mem_budget` caps the
    /// in-memory buffers used while sorting (single buckets are also capped by it).
    pub fn new(rec_size: usize, mem_budget: usize) -> Self {
        Self { rec_size, mem_budget }
    }

    pub fn rec_size(&self) -> usize {
        self.rec_size
    }

    pub fn mem_budget(&self) -> usize {
        self.mem_budget
    }

    /// Sort `input` by the u64 key returned by `key` and write the records in
    /// ascending key order to `out`. The sort is unstable: equal keys may appear in
    /// any order. Returns the number of records written.
    pub fn sort_file<K>(&self, input: &Path, out: &Path, key: K) -> io::Result<u64>
    where
        K: Fn(&[u8]) -> u64,
    {
        if self.rec_size == 0 {
            return Err(invalid_input(
                "ss-support: rec_size must be greater than zero",
            ));
        }
        let rec = self.rec_size;
        let total = fs::metadata(input)?.len();
        if total % rec as u64 != 0 {
            return Err(invalid(format!(
                "ss-support: input length {total} is not a multiple of rec_size {rec}"
            )));
        }

        // Every scratch file lives next to `out` and is removed when this guard drops,
        // so a failure halfway through leaves no bucket behind.
        let mut temps = Temps::default();
        let mut buckets: Vec<PathBuf> = Vec::with_capacity(BUCKETS);
        let route_cap = self.route_cap();
        {
            let bucket_cap = self.bucket_cap();
            let mut writers: Vec<BufWriter<File>> = Vec::with_capacity(BUCKETS);
            for index in 0..BUCKETS {
                let path = sibling(out, &format!(".b{index:03}"));
                temps.add(path.clone());
                let file = File::create(&path)?;
                buckets.push(path);
                writers.push(BufWriter::with_capacity(bucket_cap, file));
            }

            let _held = Gauge::hold(route_cap + BUCKETS * bucket_cap);
            let mut buf = vec![0u8; route_cap];
            let mut src = File::open(input)?;
            loop {
                let read = fill(&mut src, &mut buf)?;
                if read == 0 {
                    break;
                }
                for record in buf[..read - read % rec].chunks_exact(rec) {
                    let bucket = (key(record) >> 56) as usize;
                    writers[bucket].write_all(record)?;
                }
                if read < buf.len() {
                    break;
                }
            }
            for writer in writers.iter_mut() {
                writer.flush()?;
            }
        }

        let mut written = 0u64;
        {
            let mut sink = BufWriter::with_capacity(self.io_cap(), File::create(out)?);
            for path in &buckets {
                if fs::metadata(path)?.len() == 0 {
                    continue;
                }
                written += self.sort_bucket(path, &mut sink, &key, &mut temps)?;
            }
            sink.flush()?;
        }
        Ok(written)
    }

    /// Stream an already-sorted file and call `f` once per run of equal keys with
    /// the first record of the run and the run length. Bounded memory.
    pub fn scan_groups<K, F>(&self, path: &Path, key: K, f: F) -> io::Result<()>
    where
        K: Fn(&[u8]) -> u64,
        F: FnMut(&[u8], u64),
    {
        if self.rec_size == 0 {
            return Err(invalid_input(
                "ss-support: rec_size must be greater than zero",
            ));
        }
        let rec = self.rec_size;
        let len = fs::metadata(path)?.len();
        if len % rec as u64 != 0 {
            return Err(invalid(format!(
                "ss-support: file length {len} is not a multiple of rec_size {rec}"
            )));
        }

        let mut reader = BufReader::with_capacity(self.group_cap(), File::open(path)?);
        // One record of comparison state, plus the copy handed to `f` when a run ends.
        let mut current = vec![0u8; rec];
        let mut first = vec![0u8; rec];
        let mut run_key = 0u64;
        let mut run_len = 0u64;
        let mut f = f;
        for _ in 0..len / rec as u64 {
            if !fill_record(&mut reader, &mut current)? {
                return Err(invalid(format!(
                    "ss-support: {} ended inside a record",
                    path.display()
                )));
            }
            let record_key = key(&current);
            if run_len == 0 {
                first.copy_from_slice(&current);
                run_key = record_key;
                run_len = 1;
            } else if record_key == run_key {
                run_len += 1;
            } else {
                f(&first, run_len);
                first.copy_from_slice(&current);
                run_key = record_key;
                run_len = 1;
            }
        }
        if run_len > 0 {
            f(&first, run_len);
        }
        Ok(())
    }

    /// Read/write buffer for a single file handle.
    fn io_cap(&self) -> usize {
        (self.mem_budget / 32).clamp(1024, 32 * 1024)
    }

    /// Read buffer for `scan_groups`.
    fn group_cap(&self) -> usize {
        (self.mem_budget / 8).clamp(1024, 64 * 1024)
    }

    /// Record bytes one in-memory sort buffer may hold.
    fn record_cap(&self) -> usize {
        let budget = self.mem_budget.saturating_sub(2 * self.io_cap());
        (budget.min(MAX_RECORD_BUFFER) / self.rec_size).max(1) * self.rec_size
    }

    /// Record bytes read per chunk while routing.
    fn route_cap(&self) -> usize {
        let cap = (self.mem_budget / 2).min(MAX_RECORD_BUFFER);
        (cap / self.rec_size).max(1) * self.rec_size
    }

    /// Write buffer of one bucket file while routing.
    fn bucket_cap(&self) -> usize {
        self.mem_budget / (2 * BUCKETS)
    }

    /// Read buffer of one run during a merge.
    fn merge_cap(&self) -> usize {
        (self.mem_budget / (2 * MAX_FANIN)).clamp(1024, 32 * 1024)
    }

    /// Runs one merge pass may hold open: as many as half the budget feeds.
    fn fan_in(&self) -> usize {
        let per_run = self.rec_size + self.merge_cap();
        (self.mem_budget / 2 / per_run.max(1)).clamp(2, MAX_FANIN)
    }

    /// Sort one bucket into `out`: in memory when it fits, otherwise as budget-sized
    /// runs merged with a bounded fan-in. Returns the bucket's record count.
    fn sort_bucket<W, K>(
        &self,
        bucket: &Path,
        out: &mut W,
        key: &K,
        temps: &mut Temps,
    ) -> io::Result<u64>
    where
        W: Write,
        K: Fn(&[u8]) -> u64,
    {
        let rec = self.rec_size;
        let len = fs::metadata(bucket)?.len();
        let records = len / rec as u64;
        let cap = self.record_cap();

        if len <= cap as u64 {
            let _held = Gauge::hold(len as usize + self.io_cap());
            let mut buf = Vec::with_capacity(len as usize);
            File::open(bucket)?.read_to_end(&mut buf)?;
            emit_sorted(&buf, rec, key, out)?;
            return Ok(records);
        }

        let mut runs: Vec<PathBuf> = Vec::new();
        {
            let mut reader = BufReader::with_capacity(self.io_cap(), File::open(bucket)?);
            let mut buf = vec![0u8; cap];
            let mut done = 0u64;
            while done < records {
                let want = (((records - done) as usize) * rec).min(cap);
                let read = fill(&mut reader, &mut buf[..want])?;
                if read == 0 {
                    break;
                }
                let read = read - read % rec;
                let run = scratch(bucket, &format!(".r{}", runs.len()));
                {
                    let mut writer = BufWriter::with_capacity(self.io_cap(), File::create(&run)?);
                    {
                        let _held = Gauge::hold(read + 2 * self.io_cap());
                        emit_sorted(&buf[..read], rec, key, &mut writer)?;
                    }
                    writer.flush()?;
                }
                temps.add(run.clone());
                runs.push(run);
                done += (read / rec) as u64;
            }
        }

        let fan_in = self.fan_in();
        let mut pass = 0usize;
        while runs.len() > fan_in {
            let mut next = Vec::with_capacity(runs.len() / fan_in + 1);
            for (group, group_runs) in runs.chunks(fan_in).enumerate() {
                let merged = scratch(bucket, &format!(".m{pass}.{group}"));
                {
                    let mut writer = BufWriter::with_capacity(self.io_cap(), File::create(&merged)?);
                    self.merge_runs(group_runs, &mut writer, key)?;
                    writer.flush()?;
                }
                temps.add(merged.clone());
                next.push(merged);
            }
            for stale in &runs {
                let _ = fs::remove_file(stale);
            }
            runs = next;
            pass += 1;
        }
        self.merge_runs(&runs, out, key)?;
        Ok(records)
    }

    /// Merge sorted runs into `out`, holding one record per run plus bounded buffers.
    fn merge_runs<W, K>(&self, runs: &[PathBuf], out: &mut W, key: &K) -> io::Result<()>
    where
        W: Write,
        K: Fn(&[u8]) -> u64,
    {
        if runs.is_empty() {
            return Ok(());
        }
        let rec = self.rec_size;
        let cap = self.merge_cap();
        let _held = Gauge::hold(runs.len() * (rec + cap) + self.io_cap());

        let mut readers: Vec<BufReader<File>> = Vec::with_capacity(runs.len());
        for path in runs {
            readers.push(BufReader::with_capacity(cap, File::open(path)?));
        }

        let mut heads: Vec<Vec<u8>> = Vec::with_capacity(runs.len());
        let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::with_capacity(runs.len());
        for (index, reader) in readers.iter_mut().enumerate() {
            let mut head = vec![0u8; rec];
            if fill_record(reader, &mut head)? {
                heap.push(Reverse((key(&head), index)));
            } else {
                head.clear();
            }
            heads.push(head);
        }

        while let Some(Reverse((_, index))) = heap.pop() {
            let mut head = std::mem::take(&mut heads[index]);
            out.write_all(&head)?;
            head.clear();
            head.resize(rec, 0);
            if fill_record(&mut readers[index], &mut head)? {
                heap.push(Reverse((key(&head), index)));
            }
            heads[index] = head;
        }
        Ok(())
    }
}

/// Scratch files removed when dropped, so buckets never outlive the sort.
#[derive(Default)]
struct Temps {
    paths: Vec<PathBuf>,
}

impl Temps {
    fn add(&mut self, path: PathBuf) {
        self.paths.push(path);
    }
}

impl Drop for Temps {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = fs::remove_file(path);
        }
    }
}

/// Accounts record bytes held by a sorter for the lifetime of a buffer.
struct Gauge {
    bytes: usize,
}

impl Gauge {
    fn hold(bytes: usize) -> Self {
        let current = CUR_RECORD_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
        PEAK_RECORD_BYTES.fetch_max(current, Ordering::Relaxed);
        Gauge { bytes }
    }
}

impl Drop for Gauge {
    fn drop(&mut self) {
        CUR_RECORD_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// Sort the fixed-width records of `buf` by `key` and write them to `out`.
///
/// Only the permutation is held in memory, so the record bytes stay exactly as read.
fn emit_sorted<W, K>(buf: &[u8], rec: usize, key: &K, out: &mut W) -> io::Result<()>
where
    W: Write,
    K: Fn(&[u8]) -> u64,
{
    let count = buf.len() / rec;
    let mut order: Vec<u32> = (0..count as u32).collect();
    order.sort_unstable_by_key(|&index| {
        let start = index as usize * rec;
        key(&buf[start..start + rec])
    });
    for &index in order.iter() {
        let start = index as usize * rec;
        out.write_all(&buf[start..start + rec])?;
    }
    Ok(())
}

/// Fill `buf` as far as the reader allows; a short count means end of file.
fn fill<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut got = 0usize;
    while got < buf.len() {
        match reader.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

/// Fill exactly one record: `Ok(false)` at end of file, error on a partial record.
fn fill_record<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<bool> {
    let got = fill(reader, buf)?;
    if got == 0 {
        Ok(false)
    } else if got < buf.len() {
        Err(invalid("ss-support: truncated record"))
    } else {
        Ok(true)
    }
}

/// Path of the scratch file named after `base` with `suffix` appended.
fn scratch(base: &Path, suffix: &str) -> PathBuf {
    let mut name = base.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

/// Path of the file named after `out` with `suffix` appended, in `out`'s directory.
fn sibling(out: &Path, suffix: &str) -> PathBuf {
    let dir = match out.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::new(),
    };
    let mut name = match out.file_name() {
        Some(name) => name.to_os_string(),
        None => OsString::from("out"),
    };
    name.push(suffix);
    dir.join(name)
}

fn invalid_input(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
