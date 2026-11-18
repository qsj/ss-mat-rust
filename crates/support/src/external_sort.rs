//! Bounded-memory external sort for fixed-size records.
//!
//! Runs occupy consecutive ranges of two alternating scratch files. Run sizes
//! are arithmetic, so neither file names nor run descriptors grow with input.
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static CUR_RECORD_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_RECORD_BYTES: AtomicUsize = AtomicUsize::new(0);
static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

/// Peak live sort-buffer capacity, including permutations and I/O buffers.
#[doc(hidden)]
pub fn peak_buffered_record_bytes() -> usize { PEAK_RECORD_BYTES.load(Ordering::Relaxed) }
#[doc(hidden)]
pub fn reset_peak_buffered_record_bytes() {
    PEAK_RECORD_BYTES.store(CUR_RECORD_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
}

pub struct ExternalSorter { rec_size: usize, mem_budget: usize }
impl ExternalSorter {
    /// Budget covers data, permutation and read/write buffer allocations.
    /// Fixed stack state and a constant number of path strings are additional.
    pub fn new(rec_size: usize, mem_budget: usize) -> Self { Self { rec_size, mem_budget } }
    pub fn rec_size(&self) -> usize { self.rec_size }
    pub fn mem_budget(&self) -> usize { self.mem_budget }

    /// Sort by ascending key. Equal keys have an unspecified (unstable) order.
    /// Budgets too small for two records and a permutation index are rejected.
    pub fn sort_file<K>(&self, input: &Path, out: &Path, key: K) -> io::Result<u64>
    where K: Fn(&[u8]) -> u64 {
        let rec = self.rec_size;
        if rec == 0 || rec.checked_mul(2).and_then(|n| n.checked_add(4))
            .is_none_or(|minimum| minimum > self.mem_budget) {
            return Err(invalid_input("ss-support: budget must fit two records and an index"));
        }
        let mut input = File::open(input)?;
        let total = input.metadata()?.len();
        if total % rec as u64 != 0 { return Err(invalid("ss-support: truncated record")); }
        let count = total / rec as u64;
        let scratch = Scratch::new(out)?;
        let result = (|| {
        let mut source = scratch.0.join("a");
        let mut target = scratch.0.join("b");
        let io = (self.mem_budget / 8).min(32 * 1024);
        let run_records = ((self.mem_budget - io) / (rec + 4)).min(u32::MAX as usize).max(1);
        let capacity = run_records * rec;
        {
            let _held = Gauge::hold(capacity + run_records * 4 + io);
            let mut buf = vec![0; capacity];
            let mut sink = BufWriter::with_capacity(io, File::create(&source)?);
            let mut left = count;
            while left != 0 {
                let n = left.min(run_records as u64) as usize;
                input.read_exact(&mut buf[..n * rec])?;
                emit_sorted(&buf[..n * rec], rec, &key, &mut sink)?;
                left -= n as u64;
            }
            require_eof(&mut input)?;
            sink.flush()?;
        }
        drop(input);
        let mut width = run_records as u64;
        while width < count {
            self.merge_pass(&source, &target, width, count, &key)?;
            std::mem::swap(&mut source, &mut target);
            width = width.saturating_mul(self.fan_in() as u64);
        }
        // Publish only after all reads and merges succeed. The scratch is beside
        // the destination, allowing rename without a second output-sized copy.
        fs::rename(&source, out)?;
        Ok(count)
        })();
        scratch.finish(result)
    }

    fn fan_in(&self) -> usize {
        (self.mem_budget / (self.rec_size + std::mem::size_of::<BufReader<File>>() + 24)).clamp(2, 32)
    }

    fn merge_pass<K>(&self, source: &Path, target: &Path, width: u64, count: u64, key: &K) -> io::Result<()>
    where K: Fn(&[u8]) -> u64 {
        let rec = self.rec_size;
        let fan = self.fan_in();
        // At tiny budgets keep descriptors on the stack and use unbuffered reads.
        // The fixed array is stack state, not an input-sized allocation.
        let mut readers: [Option<BufReader<File>>; 32] = std::array::from_fn(|_| None);
        let cap = ((self.mem_budget - fan * rec) / (fan + 1)).min(32 * 1024);
        let _held = Gauge::hold(fan * rec + (fan + 1) * cap);
        for reader in &mut readers[..fan] {
            *reader = Some(BufReader::with_capacity(cap, File::open(source)?));
        }
        let expected = count * rec as u64;
        if fs::metadata(source)?.len() != expected { return Err(invalid("ss-support: run length changed")); }
        let mut sink = BufWriter::with_capacity(cap, File::create(target)?);
        let mut heads = vec![0; fan * rec];
        let mut remaining = [0u64; 32];
        let mut keys = [0u64; 32];
        let mut start = 0;
        while start < count {
            for i in 0..fan {
                let end = start.saturating_add(width).min(count);
                remaining[i] = end - start;
                let reader = readers[i].as_mut().unwrap();
                reader.seek(SeekFrom::Start(start * rec as u64))?;
                if remaining[i] != 0 {
                    reader.read_exact(&mut heads[i * rec..(i + 1) * rec])?;
                    keys[i] = key(&heads[i * rec..(i + 1) * rec]);
                }
                start = end;
            }
            loop {
                let winner = (0..fan).filter(|&i| remaining[i] != 0).min_by_key(|&i| keys[i]);
                let Some(i) = winner else { break };
                let head = &mut heads[i * rec..(i + 1) * rec];
                sink.write_all(head)?;
                remaining[i] -= 1;
                if remaining[i] != 0 {
                    readers[i].as_mut().unwrap().read_exact(head)?;
                    keys[i] = key(head);
                }
            }
        }
        sink.flush()?;
        if fs::metadata(source)?.len() != expected { return Err(invalid("ss-support: run length changed")); }
        Ok(())
    }
    /// Stream an already-sorted file and call `f` once per run of equal keys with
    /// the first record of the run and the run length. Bounded memory.
    pub fn scan_groups<K, F>(&self, path: &Path, key: K, f: F) -> io::Result<()>
    where
        K: Fn(&[u8]) -> u64,
        F: FnMut(&[u8], u64),
    {
        if self.rec_size == 0 || self.rec_size.checked_mul(2)
            .is_none_or(|minimum| minimum > self.mem_budget) {
            return Err(invalid_input("ss-support: group budget must fit two records"));
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
        require_eof(&mut reader)?;
        if run_len > 0 {
            f(&first, run_len);
        }
        Ok(())
    }

    fn group_cap(&self) -> usize { (self.mem_budget - 2 * self.rec_size).min(64 * 1024) }
}

struct Scratch(PathBuf);
impl Scratch {
    fn new(out: &Path) -> io::Result<Self> {
        let parent = out.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
        loop {
            let path = parent.join(format!(".ss-sort-{}-{}", std::process::id(), NEXT_TEMP.fetch_add(1, Ordering::Relaxed)));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}
impl Scratch {
    fn finish<T>(self, result: io::Result<T>) -> io::Result<T> {
        self.finish_with(result, |path| fs::remove_dir_all(path))
    }

    fn finish_with<T>(self, result: io::Result<T>, cleanup: impl FnOnce(&Path) -> io::Result<()>) -> io::Result<T> {
        match (result, cleanup(&self.0)) {
            (result, Ok(())) => result,
            (Ok(_), Err(cleanup)) => Err(io::Error::new(cleanup.kind(), format!(
                "ss-support: output published but scratch cleanup failed at {}: {cleanup}", self.0.display()
            ))),
            (Err(original), Err(cleanup)) => {
                let kind = original.kind();
                Err(io::Error::new(kind, SortCleanupError { original, cleanup, path: self.0.clone() }))
            }
        }
    }
}

#[derive(Debug)]
struct SortCleanupError { original: io::Error, cleanup: io::Error, path: PathBuf }
impl std::fmt::Display for SortCleanupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}; additionally scratch cleanup failed at {}: {}", self.original, self.path.display(), self.cleanup)
    }
}
impl std::error::Error for SortCleanupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(&self.original) }
}

impl Drop for Scratch {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}
fn require_eof<R: Read>(reader: &mut R) -> io::Result<()> {
    let mut byte = [0];
    if fill(reader, &mut byte)? != 0 { return Err(invalid("ss-support: input length changed")); }
    Ok(())
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

fn invalid_input(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidInput, message) }
fn invalid(message: impl Into<String>) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message.into()) }

#[cfg(test)]
mod cleanup_tests {
    use super::*;

    #[test]
    fn cleanup_failure_reports_path_and_preserves_original_error() {
        let out = std::env::temp_dir().join("ss-sort-cleanup-test-output");
        let scratch = Scratch::new(&out).unwrap();
        let path = scratch.0.clone();
        let err = scratch.finish_with::<()>(Err(io::Error::new(io::ErrorKind::UnexpectedEof, "original read failure")), |_| {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "injected cleanup failure"))
        }).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
        assert!(err.to_string().contains(path.to_str().unwrap()));
        let detail = err.get_ref().unwrap().downcast_ref::<SortCleanupError>().unwrap();
        assert_eq!(detail.original.to_string(), "original read failure");
        assert_eq!(detail.cleanup.kind(), io::ErrorKind::PermissionDenied);
        assert!(!path.exists(), "Drop fallback cleans the test-owned directory");

        let scratch = Scratch::new(&out).unwrap();
        let path = scratch.0.clone();
        let err = scratch.finish_with(Ok(()), |_| {
            Err(io::Error::new(io::ErrorKind::PermissionDenied, "injected cleanup failure"))
        }).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
        assert!(err.to_string().contains("output published"));
        assert!(err.to_string().contains(path.to_str().unwrap()));
        assert!(!path.exists());
    }
}
