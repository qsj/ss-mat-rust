//! Builds the on-disk object directory from the parser's single heap-dump walk.
//!
//! The directory replaces the in-memory address maps: every heap-dump object
//! becomes one 32-byte record carrying its file offset, class index, kind and
//! length, sorted by address so lookups are a binary search over a small block
//! cache instead of a hash table sized by the object count.
//!
//! The records are fed straight from the phase-1 walk (see
//! [ObjDirSink]), so the dump is never traversed a second time just to index it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use ss_support::obj_dir::{ObjDir, ObjDirBuilder, ObjRec, OBJ_REC_SIZE};

/// Object kind of a class dump, matching [ss_support::obj_dir::KIND_CLASS_DUMP].
pub const KIND_CLASS_DUMP: u8 = ss_support::obj_dir::KIND_CLASS_DUMP;

/// Default page cache for the object directory.
pub const DEFAULT_CACHE_BYTES: usize = 32 << 20;

/// Where temporary files go and how much room they may use.
#[derive(Clone, Debug)]
pub struct ObjDirConfig {
    pub temp_dir: PathBuf,
    pub cache_bytes: usize,
    pub disk_budget_bytes: u64,
}

impl Default for ObjDirConfig {
    fn default() -> Self {
        let temp_dir = std::env::var_os("SS_MAT_TEMP_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join(format!("ss-mat-{}", std::process::id())));
        Self {
            temp_dir,
            cache_bytes: DEFAULT_CACHE_BYTES,
            disk_budget_bytes: 4 << 30,
        }
    }
}

/// A temporary working directory that removes itself on drop.
pub struct TempWorkspace {
    path: PathBuf,
    owned: bool,
}

impl TempWorkspace {
    /// Create (or adopt) the working directory. A directory this run created is
    /// removed again on drop; one that already existed is left alone.
    pub fn create(path: &Path) -> Result<Self> {
        let owned = !path.exists();
        std::fs::create_dir_all(path)
            .with_context(|| format!("cannot create temp dir {}", path.display()))?;
        Ok(Self { path: path.to_path_buf(), owned })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempWorkspace {
    fn drop(&mut self) {
        if self.owned {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Rough space the directory needs: one record per object plus bucket spill.
pub fn estimate_disk_bytes(object_count: u64) -> u64 {
    object_count.saturating_mul(OBJ_REC_SIZE as u64 * 2) + (64 << 20)
}

/// Incremental object-directory writer.
///
/// One object per [ObjDirSink::push], in dump order; [ObjDirSink::finish] sorts
/// the records, writes the block index and opens the directory for lookups.
pub struct ObjDirSink {
    builder: ObjDirBuilder,
    records: u64,
}

impl ObjDirSink {
    /// Start a directory under the configured temp directory.
    pub fn new(temp_dir: &Path) -> Result<Self> {
        let builder = ObjDirBuilder::new(temp_dir)
            .with_context(|| format!("cannot create the object directory in {}", temp_dir.display()))?;
        Ok(Self { builder, records: 0 })
    }

    /// Append one object record.
    ///
    /// file_offset is the object's body offset in the dump (for a class dump, the
    /// start of its class-dump body, i.e. the byte after the 0x20 tag).
    pub fn push(
        &mut self,
        addr: u64,
        file_offset: u64,
        class_idx: u32,
        len: u32,
        kind: u8,
        elem: u8,
        flags: u16,
    ) -> Result<()> {
        self.builder
            .push(ObjRec { addr, file_offset, class_idx, len, kind, elem, flags })
            .with_context(|| format!("failed to append object record for {addr:#x}"))?;
        self.records += 1;
        Ok(())
    }

    /// Number of records appended so far.
    pub fn records(&self) -> u64 {
        self.records
    }

    /// Close the directory and open it for lookups.
    pub fn finish(self, cfg: &ObjDirConfig) -> Result<ObjDir> {
        let need = estimate_disk_bytes(self.records);
        if need > cfg.disk_budget_bytes {
            anyhow::bail!(
                "object directory needs about {} MB but the disk budget is {} MB; raise the budget or point the temp dir at a larger volume",
                need >> 20,
                cfg.disk_budget_bytes >> 20
            );
        }
        let file = cfg.temp_dir.join("objects.objdir");
        self.builder
            .finish(&file, cfg.cache_bytes)
            .with_context(|| format!("failed to finish the object directory at {}", file.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_scales_with_objects() {
        assert!(estimate_disk_bytes(6_000_000) > estimate_disk_bytes(1_000_000));
    }
}
