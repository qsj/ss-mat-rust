//! On-disk object directory: address -> (file offset, class, kind, length).
//!
//! Replaces the in-memory address maps the pipeline used to keep. Records are a
//! fixed 32 bytes, grouped into 128 KiB blocks, sorted by address, and looked up
//! through a coarse index plus a bounded block cache, so resident memory is the
//! cache budget rather than a hash table proportional to the object count.
//!
//! File layout:
//!   [records][coarse addrs: u64 * blocks][n: u64][magic: u64]
//! The i-th coarse entry is the address of record i * BLOCK_RECS; the footer lets
//! open() rebuild the index without rescanning the records.

use std::fs::{File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;

/// Object kind for a class-dump record (instances are 0, primitive arrays 1, object arrays 2).
pub const KIND_CLASS_DUMP: u8 = 3;

/// Fixed on-disk record width.
pub const OBJ_REC_SIZE: usize = 32;

/// Records per cache block.
pub const BLOCK_RECS: u64 = 4096;

/// Block size in bytes.
pub const BLOCK_BYTES: usize = BLOCK_RECS as usize * OBJ_REC_SIZE;

const MAGIC: u64 = 0x5353_4f42_4a44_4952;

/// Sort data/permutation/I/O budget; fixed metadata and paths are additional.
const SORT_BUFFER_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjRec {
    pub addr: u64,
    pub file_offset: u64,
    pub class_idx: u32,
    pub len: u32,
    pub kind: u8,
    pub elem: u8,
    pub flags: u16,
}

impl ObjRec {
    pub fn to_bytes(self) -> [u8; OBJ_REC_SIZE] {
        let mut out = [0u8; OBJ_REC_SIZE];
        out[0..8].copy_from_slice(&self.addr.to_le_bytes());
        out[8..16].copy_from_slice(&self.file_offset.to_le_bytes());
        out[16..20].copy_from_slice(&self.class_idx.to_le_bytes());
        out[20..24].copy_from_slice(&self.len.to_le_bytes());
        out[24] = self.kind;
        out[25] = self.elem;
        out[26..28].copy_from_slice(&self.flags.to_le_bytes());
        out
    }

    pub fn from_bytes(b: &[u8]) -> Self {
        let mut addr = [0u8; 8];
        addr.copy_from_slice(&b[0..8]);
        let mut off = [0u8; 8];
        off.copy_from_slice(&b[8..16]);
        let mut cls = [0u8; 4];
        cls.copy_from_slice(&b[16..20]);
        let mut len = [0u8; 4];
        len.copy_from_slice(&b[20..24]);
        let mut flags = [0u8; 2];
        flags.copy_from_slice(&b[26..28]);
        Self {
            addr: u64::from_le_bytes(addr),
            file_offset: u64::from_le_bytes(off),
            class_idx: u32::from_le_bytes(cls),
            len: u32::from_le_bytes(len),
            kind: b[24],
            elem: b[25],
            flags: u16::from_le_bytes(flags),
        }
    }
}

struct Slot {
    block: u64,
    valid: bool,
    data: Vec<u8>,
}

struct BlockCache {
    slots: Vec<Slot>,
    mask: u64,
}

impl BlockCache {
    fn new(cache_bytes: usize) -> Self {
        let target = (cache_bytes / BLOCK_BYTES).max(1);
        let mut slots = 1usize;
        while slots * 2 <= target {
            slots *= 2;
        }
        Self {
            slots: (0..slots)
                .map(|_| Slot { block: u64::MAX, valid: false, data: Vec::new() })
                .collect(),
            mask: (slots - 1) as u64,
        }
    }

    fn slot_for(&mut self, block: u64) -> &mut Slot {
        let idx = (block & self.mask) as usize;
        &mut self.slots[idx]
    }
}

/// Read-only view of an on-disk object directory.
pub struct ObjDir {
    file: File,
    path: PathBuf,
    len: u64,
    coarse: Vec<u64>,
    cache: Mutex<BlockCache>,
}

impl ObjDir {
    /// Open an existing directory file, sizing the block cache at cache_bytes.
    pub fn open(path: &Path, cache_bytes: usize) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir file too short"));
        }
        file.seek(SeekFrom::Start(file_len - 16))?;
        let mut tail = [0u8; 16];
        file.read_exact(&mut tail)?;
        let mut n_buf = [0u8; 8];
        n_buf.copy_from_slice(&tail[0..8]);
        let mut m_buf = [0u8; 8];
        m_buf.copy_from_slice(&tail[8..16]);
        let n = u64::from_le_bytes(n_buf);
        let magic = u64::from_le_bytes(m_buf);
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir magic mismatch"));
        }
        let blocks = n.div_ceil(BLOCK_RECS);
        let coarse_bytes = blocks * 8;
        let expected = n.checked_mul(OBJ_REC_SIZE as u64)
            .and_then(|bytes| bytes.checked_add(coarse_bytes))
            .and_then(|bytes| bytes.checked_add(16));
        if expected != Some(file_len) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir truncated"));
        }
        let coarse_off = file_len - 16 - coarse_bytes;
        let mut buf = vec![0u8; coarse_bytes as usize];
        file.seek(SeekFrom::Start(coarse_off))?;
        file.read_exact(&mut buf)?;
        let coarse: Vec<u64> = buf
            .chunks_exact(8)
            .map(|c| {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                u64::from_le_bytes(b)
            })
            .collect();
        Ok(Self { file, path: path.to_path_buf(), len: n, coarse, cache: Mutex::new(BlockCache::new(cache_bytes)) })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of records.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read record idx (0-based, in address order).
    pub fn get(&self, idx: u64) -> io::Result<ObjRec> {
        if idx >= self.len {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "objdir index out of range"));
        }
        let block = idx / BLOCK_RECS;
        let off_in_block = (idx % BLOCK_RECS) as usize * OBJ_REC_SIZE;
        let mut cache = self.cache.lock();
        let slot = cache.slot_for(block);
        if !slot.valid || slot.block != block {
            self.fill_block(block, slot)?;
        }
        Ok(ObjRec::from_bytes(&slot.data[off_in_block..off_in_block + OBJ_REC_SIZE]))
    }

    /// Binary-search addr, returning its record index and record.
    pub fn lookup(&self, addr: u64) -> io::Result<Option<(u64, ObjRec)>> {
        if self.len == 0 || self.coarse.is_empty() {
            return Ok(None);
        }
        let mut lo = 0usize;
        let mut hi = self.coarse.len();
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.coarse[mid] <= addr {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let block = lo as u64;
        let recs_in_block = std::cmp::min(BLOCK_RECS, self.len - block * BLOCK_RECS);
        let hit = {
            let mut cache = self.cache.lock();
            let slot = cache.slot_for(block);
            if !slot.valid || slot.block != block {
                self.fill_block(block, slot)?;
            }
            let mut bl = 0u64;
            let mut bh = recs_in_block;
            while bl < bh {
                let mid = (bl + bh) / 2;
                let off = mid as usize * OBJ_REC_SIZE;
                let rec = ObjRec::from_bytes(&slot.data[off..off + OBJ_REC_SIZE]);
                if rec.addr < addr {
                    bl = mid + 1;
                } else {
                    bh = mid;
                }
            }
            if bl < recs_in_block {
                let off = bl as usize * OBJ_REC_SIZE;
                let rec = ObjRec::from_bytes(&slot.data[off..off + OBJ_REC_SIZE]);
                if rec.addr == addr {
                    Some((block * BLOCK_RECS + bl, rec))
                } else {
                    None
                }
            } else {
                None
            }
        };
        Ok(hit)
    }

    /// True when addr has a record.
    pub fn contains(&self, addr: u64) -> io::Result<bool> {
        Ok(self.lookup(addr)?.is_some())
    }

    fn fill_block(&self, block: u64, slot: &mut Slot) -> io::Result<()> {
        let start = block * BLOCK_RECS;
        let recs = std::cmp::min(BLOCK_RECS, self.len - start);
        let bytes = recs as usize * OBJ_REC_SIZE;
        slot.data.resize(bytes, 0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(&mut slot.data[..bytes], start * OBJ_REC_SIZE as u64)?;
        }
        #[cfg(not(unix))]
        {
            let mut f = self.file.try_clone()?;
            f.seek(SeekFrom::Start(start * OBJ_REC_SIZE as u64))?;
            f.read_exact(&mut slot.data[..bytes])?;
        }
        slot.block = block;
        slot.valid = true;
        Ok(())
    }
}

/// Spills records, externally sorts, and streams coarse anchors.
pub struct ObjDirBuilder {
    dir: PathBuf,
    spill: Option<BufWriter<File>>,
    pushed: u64,
}
impl ObjDirBuilder {
    pub fn new(temp_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(temp_dir)?;
        let dir = temp_dir.join(format!("objdir-{}-{}", std::process::id(), now_nanos()));
        std::fs::create_dir(&dir)?;
        let mut b = Self { dir, spill: None, pushed: 0 };
        b.spill = Some(BufWriter::new(File::create(b.dir.join("spill"))?));
        Ok(b)
    }
    pub fn push(&mut self, rec: ObjRec) -> io::Result<()> {
        self.spill.as_mut().expect("spill writer").write_all(&rec.to_bytes())?;
        self.pushed += 1;
        Ok(())
    }
    pub fn len(&self) -> u64 { self.pushed }
    pub fn is_empty(&self) -> bool { self.pushed == 0 }
    pub fn finish(self, out: &Path, cache: usize) -> io::Result<ObjDir> {
        self.finish_with_sort_budget(out, cache, SORT_BUFFER_BYTES)
    }
    /// Override buffer capacity; fixed sorter/path overhead is additional.
    #[doc(hidden)]
    pub fn finish_with_sort_budget(mut self, out: &Path, cache: usize, budget: usize) -> io::Result<ObjDir> {
        let result = (|| {
            if let Some(mut w) = self.spill.take() { w.flush()?; }
            let spill = self.dir.join("spill");
            let expected = self.pushed.checked_mul(OBJ_REC_SIZE as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "objdir count overflow"))?;
            if std::fs::metadata(&spill)?.len() != expected {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir spill length/count mismatch"));
            }
            let sorted = self.dir.join("sorted");
            let count = crate::external_sort::ExternalSorter::new(OBJ_REC_SIZE, budget)
                .sort_file(&spill, &sorted, |r| ObjRec::from_bytes(r).addr)?;
            if count != self.pushed || std::fs::metadata(&sorted)?.len() != expected {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir sorted length/count mismatch"));
            }
            let anchors = self.dir.join("anchors");
            {
                let mut input = std::io::BufReader::new(File::open(&sorted)?);
                let mut coarse = BufWriter::new(File::create(&anchors)?);
                let mut bytes = [0; OBJ_REC_SIZE];
                let mut previous = None;
                for i in 0..count {
                    input.read_exact(&mut bytes)?;
                    let addr = ObjRec::from_bytes(&bytes).addr;
                    if previous == Some(addr) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData,
                            format!("objdir duplicate address 0x{addr:016x}")));
                    }
                    if previous.is_some_and(|p| p > addr) {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir addresses out of order"));
                    }
                    previous = Some(addr);
                    if i % BLOCK_RECS == 0 { coarse.write_all(&addr.to_le_bytes())?; }
                }
                if input.read(&mut bytes[..1])? != 0 {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir sorted trailing bytes"));
                }
                coarse.flush()?;
            }
            {
                let mut output = BufWriter::new(std::fs::OpenOptions::new().append(true).open(&sorted)?);
                let mut input = File::open(&anchors)?;
                let mut buf = [0; 8192];
                loop { let n = input.read(&mut buf)?; if n == 0 { break; } output.write_all(&buf[..n])?; }
                output.write_all(&count.to_le_bytes())?;
                output.write_all(&MAGIC.to_le_bytes())?;
                output.flush()?;
            }
            publish_directory(&sorted, out)?;
            Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(&self.dir);
        match (result, cleanup) {
            (Err(original), Err(cleanup)) => return Err(io::Error::new(original.kind(),
                CleanupError { original, cleanup, path: self.dir.clone() })),
            (Err(original), _) => return Err(original),
            (_, Err(cleanup)) => return Err(io::Error::new(cleanup.kind(),
                format!("objdir output published but scratch cleanup failed at {}: {cleanup}", self.dir.display()))),
            _ => {}
        }
        ObjDir::open(out, cache)
    }
}
impl Drop for ObjDirBuilder {
    fn drop(&mut self) {
        self.spill.take();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[derive(Debug)]
struct CleanupError { original: io::Error, cleanup: io::Error, path: PathBuf }
impl std::fmt::Display for CleanupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}; objdir cleanup failed at {}: {}", self.original, self.path.display(), self.cleanup)
    }
}
impl std::error::Error for CleanupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { Some(&self.original) }
}

fn publish_directory(source: &Path, destination: &Path) -> io::Result<()> {
    match std::fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::CrossesDevices => stage_directory(source, destination, |src, dst| {
            let mut buf = [0; 8192];
            loop {
                let n = src.read(&mut buf)?;
                if n == 0 { break; }
                dst.write_all(&buf[..n])?;
            }
            Ok(())
        }),
        Err(e) => Err(e),
    }
}

struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}
fn stage_directory(source: &Path, destination: &Path,
    copy: impl FnOnce(&mut File, &mut File) -> io::Result<()>) -> io::Result<()> {
    let parent = destination.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let (staging, mut output) = loop {
        let path = parent.join(format!(".objdir-stage-{}-{}", std::process::id(), now_nanos()));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => break (Staging(path), file),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let result = (|| {
        let mut input = File::open(source)?;
        let expected = input.metadata()?.len();
        copy(&mut input, &mut output)?;
        output.flush()?;
        if output.metadata()?.len() != expected || input.metadata()?.len() != expected {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "objdir publication length changed"));
        }
        drop(output);
        std::fs::rename(&staging.0, destination)
    })();
    match result {
        Ok(()) => Ok(()),
        Err(original) => match std::fs::remove_file(&staging.0) {
            Ok(()) => Err(original),
            Err(cleanup) => Err(io::Error::new(original.kind(), CleanupError {
                original, cleanup, path: staging.0.clone()
            })),
        },
    }
}

fn now_nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(addr: u64, off: u64) -> ObjRec {
        ObjRec {
            addr,
            file_offset: off,
            class_idx: (off % 97) as u32,
            len: (off % 13) as u32,
            kind: (off % 4) as u8,
            elem: 1,
            flags: 0,
        }
    }

    #[test]
    fn staged_publication_preserves_old_output_and_cleans_failures() {
        let dir = std::env::temp_dir().join(format!("objdir-stage-test-{}-{}", std::process::id(), now_nanos()));
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("source"); let out = dir.join("out");
        std::fs::write(&source, b"new directory").unwrap();
        std::fs::write(&out, b"old directory").unwrap();
        let err = stage_directory(&source, &out, |_, dst| {
            dst.write_all(b"partial")?;
            Err(io::Error::new(io::ErrorKind::Other, "injected copy failure"))
        }).unwrap_err();
        assert!(err.to_string().contains("injected copy failure"));
        assert_eq!(std::fs::read(&out).unwrap(), b"old directory");
        assert!(stage_directory(&source, &out, |_, dst| dst.write_all(b"short")).is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"old directory");
        stage_directory(&source, &out, |src, dst| { io::copy(src, dst)?; Ok(()) }).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"new directory");
        assert!(std::fs::read_dir(&dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().starts_with(".objdir-stage-")));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn roundtrip_and_lookup() {
        let dir = std::env::temp_dir().join(format!("objdir-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut b = ObjDirBuilder::new(&dir).unwrap();
        let mut addrs: Vec<u64> = (0..10_000u64).map(|i| ((i * 7919) << 13) | (i & 0xfff)).collect();
        addrs.sort_unstable();
        for (i, &a) in addrs.iter().enumerate() {
            b.push(rec(a, i as u64 * 64)).unwrap();
        }
        let out = dir.join("objects.objdir");
        let d = b.finish(&out, 1 << 20).unwrap();
        assert_eq!(d.len(), addrs.len() as u64);
        for (i, &a) in addrs.iter().enumerate() {
            let (idx, r) = d.lookup(a).unwrap().expect("present");
            assert_eq!(idx, i as u64);
            assert_eq!(r.addr, a);
            assert_eq!(r.file_offset, i as u64 * 64);
            assert_eq!(d.get(idx).unwrap(), r);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
