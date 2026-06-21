# SS-Mat (Rust)

A JVM heap dump (.hprof) analyzer that detects memory problems and produces an
HTML report. A Rust port of [JXRay](https://github.com/google/jxray)-style
analysis, tuned to match the original Java SS-Mat tool's report format and
data values.

## Quick start

```bash
cargo build --release
./target/release/ss-mat <heap_dump.hprof> [report.html]
```

Then open the generated `report.html` in a browser.

## Requirements

- Rust 1.70+ (2021 edition)
- The heap dump must be a **decompressed** `.hprof` file (compress it yourself
  if needed — `.gz`/`.zip` inputs are rejected with a clear message)

## What it detects

All 24 report sections are populated, including:

- **DirectByteBuffer off-heap memory** — address-deduplicated capacity, with a
  reachable/unreachable split (the "easy-to-fix" subset)
- **Duplicate strings** — Java 8 (char[]) and Java 9+ (byte[], compact strings)
- **Bad collections** — empty / single-element / sparse (capacity ≫ size)
- **Bad object/primitive arrays** — empty, sparse, trailing-zeros
- **Boxed numbers, duplicate objects/arrays, humongous objects** (>½ G1 region)
- **OOM thread** — stack trace with decoded local variable values
- **Per-class memory breakdown, live vs garbage, GC-root tree, thread stacks**

## CLI options

```
ss-mat <heap_dump.hprof> [report.html] [options]

Options:
  --min-reported-percent <PCT>       Min heap % to report a problem (default 0.1)
  --min-trailing-zeroes-percent <P>  Trailing-zero threshold (default 25.0)
  --min-sparse-zeroes-percent <P>    Sparse-array threshold (default 70.0)
  --jvm-flags-file <PATH>           External JVM flags file
  --max-scan-threads <N>            Object-graph scan threads (0 = auto)
  --hprof-pointer-size <4|8>        Force HPROF id size (0 = auto-detect)
```

Run `ss-mat` with no arguments for full help.

## Architecture

8-crate workspace:

| Crate | Role |
|---|---|
| `hprof-parser` | HPROF binary parsing → `HeapDump` model |
| `heap-model` | Object/class/field data model |
| `traversal` | Linear scan (field extraction) + BFS reachability traversal |
| `stats` | Problem-detection collectors + per-class stats |
| `report` | HTML report generation (24 sections + encoded-string dedup) |
| `containers` | Collection descriptors (HashMap/ArrayList/CHM/...) |
| `support` | Internal data structures |
| `cli` | Entry point, pipeline orchestration, error handling |

**Pipeline**: parse HPROF → build index → re-parse for field data → parallel
linear scan + BFS traversal → problem detection → HTML report.

## Running tests

```bash
cargo test
```

Core size calculations and detection logic are covered by unit tests in
`crates/traversal/src/linear_scan.rs`.

## Error handling

- Invalid/truncated HPROF files produce a clear message (not a raw backtrace)
- Non-existent, compressed, or too-small files are rejected up front
- Failures exit with a non-zero status code
