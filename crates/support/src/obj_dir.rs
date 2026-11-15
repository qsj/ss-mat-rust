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

/// Build bucket count; buckets are selected by the high byte of the address.
const BUCKETS: usize = 256;

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
        if file_len < 16 + coarse_bytes {
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

/// Building side: records spill to one sequential file, then a sample of the
/// addresses picks 256 range boundaries so every bucket holds roughly the same
/// number of records. Bucketing by the high address byte is not usable here —
/// JVM heaps keep every object inside a narrow address range, so one bucket would
/// swallow the whole dump.
pub struct ObjDirBuilder {
    dir: PathBuf,
    spill: Option<BufWriter<File>>,
    pushed: u64,
    sample: Vec<u64>,
    sample_stride: u64,
}

const SAMPLE_TARGET: usize = 8192;

impl ObjDirBuilder {
    pub fn new(temp_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(temp_dir)?;
        let dir = temp_dir.join(format!("objdir-{}-{}", std::process::id(), now_nanos()));
        std::fs::create_dir_all(&dir)?;
        let spill = File::create(dir.join("spill"))?;
        Ok(Self {
            dir,
            spill: Some(BufWriter::new(spill)),
            pushed: 0,
            sample: Vec::with_capacity(SAMPLE_TARGET),
            sample_stride: 1,
        })
    }

    /// Append one record to the spill file and feed the address sample.
    pub fn push(&mut self, rec: ObjRec) -> io::Result<()> {
        let w = self.spill.as_mut().expect("spill writer");
        w.write_all(&rec.to_bytes())?;
        self.pushed += 1;
        Ok(())
    }

    pub fn len(&self) -> u64 {
        self.pushed
    }

    pub fn is_empty(&self) -> bool {
        self.pushed == 0
    }

    /// Address sample taken uniformly by record position over the spill file.
    ///
    /// The push-time reservoir thinned with a growing stride, which is wrong as soon
    /// as addresses are monotone (a real dump is): the retained sample covered only
    /// the first fraction of the range, every quantile bound collapsed onto the same
    /// value, and one bucket absorbed the whole dump. Position-uniform sampling has
    /// no such bias.
    fn uniform_sample(&self, spill: &Path) -> io::Result<Vec<u64>> {
        if self.pushed == 0 {
            return Ok(Vec::new());
        }
        let stride = (self.pushed / SAMPLE_TARGET as u64).max(1);
        let mut out: Vec<u64> = Vec::with_capacity(SAMPLE_TARGET + 1);
        let mut reader = std::io::BufReader::new(File::open(spill)?);
        let mut rec = [0u8; OBJ_REC_SIZE];
        let mut i: u64 = 0;
        loop {
            match reader.read_exact(&mut rec) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            if i % stride == 0 {
                out.push(ObjRec::from_bytes(&rec).addr);
            }
            i += 1;
        }
        Ok(out)
    }

    fn boundaries_from(&self, mut s: Vec<u64>) -> Vec<u64> {
        if s.is_empty() {
            return vec![0; BUCKETS + 1];
        }
        s.sort_unstable();
        let mut out = Vec::with_capacity(BUCKETS + 1);
        for i in 0..=BUCKETS {
            let pos = (i * s.len()) / BUCKETS;
            out.push(if pos == 0 { 0 } else { s[(pos - 1).min(s.len() - 1)] });
        }
        out
    }

    fn bucket_of(bounds: &[u64], addr: u64) -> usize {
        match bounds.binary_search(&addr) {
            Ok(i) => i.min(BUCKETS - 1),
            Err(i) => i.saturating_sub(1).min(BUCKETS - 1),
        }
    }

    /// Partition, sort and concatenate, then open the finished directory.
    pub fn finish(mut self, out_path: &Path, cache_bytes: usize) -> io::Result<ObjDir> {
        if let Some(mut w) = self.spill.take() {
            w.flush()?;
        }
        let spill_path = self.dir.join("spill");
        let bounds = self.boundaries_from(self.uniform_sample(&spill_path)?);

        // 1. Range-partition the spill file.
        let mut writers: Vec<Option<BufWriter<File>>> = (0..BUCKETS).map(|_| None).collect();
        let mut counts = vec![0u64; BUCKETS];
        {
            let mut input = std::io::BufReader::new(File::open(&spill_path)?);
            let mut rec = [0u8; OBJ_REC_SIZE];
            loop {
                match input.read_exact(&mut rec) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                    Err(e) => return Err(e),
                }
                let addr = ObjRec::from_bytes(&rec).addr;
                let b = Self::bucket_of(&bounds, addr);
                if writers[b].is_none() {
                    writers[b] = Some(BufWriter::new(File::create(self.dir.join(format!("b{:03}", b)))?));
                }
                writers[b].as_mut().expect("bucket writer").write_all(&rec)?;
                counts[b] += 1;
            }
        }
        for w in writers.iter_mut().flatten() {
            w.flush()?;
        }
        drop(writers);
        let _ = std::fs::remove_file(&spill_path);

        // 2. Sort each bucket in memory (one bucket at a time) and concatenate.
        let mut out = BufWriter::new(File::create(out_path)?);
        let mut coarse: Vec<u64> = Vec::with_capacity((self.pushed / BLOCK_RECS) as usize + 2);
        let mut written: u64 = 0;
        for bucket in 0..BUCKETS {
            if counts[bucket] == 0 {
                continue;
            }
            let path = self.dir.join(format!("b{:03}", bucket));
            let mut raw = Vec::with_capacity(counts[bucket] as usize * OBJ_REC_SIZE);
            File::open(&path)?.read_to_end(&mut raw)?;
            if raw.len() % OBJ_REC_SIZE != 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "bucket size not a multiple of record size"));
            }
            let mut recs: Vec<ObjRec> = raw.chunks_exact(OBJ_REC_SIZE).map(ObjRec::from_bytes).collect();
            raw = Vec::new();
            recs.sort_unstable_by_key(|r| r.addr);
            let mut buf: Vec<u8> = Vec::with_capacity(recs.len() * OBJ_REC_SIZE);
            for rec in &recs {
                if written % BLOCK_RECS == 0 {
                    coarse.push(rec.addr);
                }
                buf.extend_from_slice(&rec.to_bytes());
                written += 1;
            }
            out.write_all(&buf)?;
            drop(recs);
            let _ = std::fs::remove_file(&path);
        }

        for addr in &coarse {
            out.write_all(&addr.to_le_bytes())?;
        }
        out.write_all(&written.to_le_bytes())?;
        out.write_all(&MAGIC.to_le_bytes())?;
        out.flush()?;
        let _ = std::fs::remove_dir_all(&self.dir);

        ObjDir::open(out_path, cache_bytes)
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
