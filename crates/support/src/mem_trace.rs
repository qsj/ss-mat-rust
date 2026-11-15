//! Optional memory tracing: phase-boundary RSS snapshots plus an allocation counter.
//!
//! Enabled by SS_MAT_MEM_TRACE=1 (or a value). Inert otherwise: the allocator hooks
//! are two relaxed atomic operations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static FREES: AtomicU64 = AtomicU64::new(0);
static PEAK_RSS_KB: AtomicU64 = AtomicU64::new(0);

/// Wraps the system allocator so the pipeline can report live/peak heap bytes.
pub struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            note_alloc(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc_zeroed(layout);
        if !p.is_null() {
            note_alloc(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        FREES.fetch_add(1, Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = System.realloc(ptr, layout, new_size);
        if !p.is_null() {
            LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
            note_alloc(new_size);
        }
        p
    }
}

#[inline]
fn note_alloc(size: usize) {
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    ALLOCS.fetch_add(1, Ordering::Relaxed);
    let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
    while live > peak {
        match PEAK_BYTES.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => peak = observed,
        }
    }
}

/// True when SS_MAT_MEM_TRACE is set to anything but "0"/"".
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("SS_MAT_MEM_TRACE") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    })
}

fn rss_kb() -> u64 {
    if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
        if let Some(pages) = s.split_whitespace().nth(1) {
            if let Ok(p) = pages.parse::<u64>() {
                return p * 4;
            }
        }
    }
    0
}

fn anon_kb() -> u64 {
    if let Ok(s) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("Anonymous:") {
                if let Some(kb) = rest.split_whitespace().next() {
                    return kb.parse::<u64>().unwrap_or(0);
                }
            }
        }
    }
    0
}

/// Print a phase-boundary memory line and update the sampled RSS peak.
pub fn trace(label: &str) {
    let rss = rss_kb();
    PEAK_RSS_KB.fetch_max(rss, Ordering::Relaxed);
    if !enabled() {
        return;
    }
    let live = LIVE_BYTES.load(Ordering::Relaxed);
    eprintln!(
        "[mem] {:<28} rss={:>7}KB anon={:>7}KB live={:>7}KB peak_live={:>7}KB allocs={}",
        label,
        rss,
        anon_kb(),
        live / 1024,
        PEAK_BYTES.load(Ordering::Relaxed) / 1024,
        ALLOCS.load(Ordering::Relaxed)
    );
}

/// One-line peak summary.
pub fn summary() {
    if !enabled() {
        return;
    }
    eprintln!(
        "[mem] peak: rss={}KB live_heap={}KB allocs={} frees={}",
        PEAK_RSS_KB.load(Ordering::Relaxed),
        PEAK_BYTES.load(Ordering::Relaxed) / 1024,
        ALLOCS.load(Ordering::Relaxed),
        FREES.load(Ordering::Relaxed)
    );
}

/// Current live heap bytes (allocator accounting).
pub fn live_bytes() -> usize {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// Peak live heap bytes since start.
pub fn peak_bytes() -> usize {
    PEAK_BYTES.load(Ordering::Relaxed)
}
