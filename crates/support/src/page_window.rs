//! Bounded-resident access to large files.
//!
//! Sequential scans use a whole-file read-only mapping and release the pages the
//! cursor has already passed (`MADV_DONTNEED`), so the resident set stays near the
//! window size instead of the file size. Random reads use `pread` into a caller
//! buffer, which never leaves file pages attributed to this process.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

use memmap2::{Mmap, UncheckedAdvice};

/// Granularity the release window is rounded up to.
const PAGE_BYTES: usize = 4096;

/// Smallest release window, per the API contract.
const MIN_WINDOW_BYTES: usize = 1 << 20;

/// A file opened for windowed sequential scans plus random reads.
pub struct PageWindowFile {
    file: File,
    map: Option<Mmap>,
    len: u64,
    window_bytes: usize,
}

impl PageWindowFile {
    /// Map `path` read-only. `window_bytes` is the release granularity used by
    /// [`PageWindowFile::scan_seq`]; it is rounded up to a page multiple and
    /// clamped to at least 1 MiB.
    pub fn open(path: &Path, window_bytes: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        let window_bytes = window_bytes.max(MIN_WINDOW_BYTES);
        let window_bytes = window_bytes
            .checked_next_multiple_of(PAGE_BYTES)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "ss-support: window_bytes overflows a page-rounded window",
                )
            })?;
        // SAFETY: the mapping is only ever read through `&[u8]`; a concurrent writer
        // to the same path is outside this type's contract, as for any mmap reader.
        let map = if len == 0 {
            None
        } else {
            Some(unsafe { Mmap::map(&file)? })
        };
        Ok(Self {
            file,
            map,
            len,
            window_bytes,
        })
    }

    /// Total file length in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Visit the file sequentially. `f` is called with `(offset, chunk)` for
    /// consecutive, non-overlapping chunks covering the whole file; each chunk is at
    /// most `chunk_bytes` long and borrows the mapping, so it is only valid for the
    /// duration of the call. Pages more than `window_bytes` behind the cursor are
    /// released as the scan advances, which keeps RSS bounded by the window.
    pub fn scan_seq<F>(&mut self, chunk_bytes: usize, f: F) -> io::Result<()>
    where
        F: FnMut(u64, &[u8]),
    {
        if chunk_bytes == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ss-support: scan_seq chunk_bytes must be greater than zero",
            ));
        }
        let map = match self.map.as_ref() {
            Some(map) => map,
            // An empty file has no mapping; the callback is never invoked.
            None => return Ok(()),
        };
        let mut f = f;
        let len = map.len();
        let window = self.window_bytes.max(PAGE_BYTES);
        let mut released = 0usize;
        let mut offset = 0usize;
        while offset < len {
            let end = (offset + chunk_bytes).min(len);
            f(offset as u64, &map[offset..end]);
            offset = end;
            // Everything that is now more than one window behind the cursor is dead
            // weight. Advising costs a syscall per window, so only do it once enough
            // new pages have been passed to fill a window.
            let target = offset.saturating_sub(window);
            if target >= released + window {
                // SAFETY: the range lies strictly behind the scan cursor, so dropping
                // its pages is sound here; a later re-read just faults them back in
                // from the read-only file mapping.
                unsafe {
                    map.unchecked_advise_range(
                        UncheckedAdvice::DontNeed,
                        released,
                        target - released,
                    )?;
                }
                released = target;
            }
        }
        Ok(())
    }

    /// Random access: copy bytes starting at `offset` into `buf` using `pread`,
    /// returning the number of bytes copied (short only at end of file).
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut done = 0usize;
        while done < buf.len() {
            match self.file.read_at(&mut buf[done..], offset + done as u64) {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(done)
    }
}
