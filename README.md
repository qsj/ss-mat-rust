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
- **Bad primitive arrays** — 0/1-length, all-zero, trailing-zeroes, sparse,
  unused-high-bits (matching Java's six categories)
- **Bad object arrays** — empty, single-element, all-null, one-nonnull, sparse
- **Boxed numbers, duplicate objects/arrays, humongous objects** (>½ G1 region)
- **OOM thread** — stack trace with decoded local variable values
- **Per-class memory breakdown, live vs garbage, GC-root tree, thread stacks**

## CLI options

```
ss-mat <heap_dump.hprof> [report.html] [options]

Options:
  --min-trailing-zeroes-percent <P>  Trailing-zero threshold (default 25.0)
  --min-sparse-zeroes-percent <P>    Sparse-array threshold (default 70.0)
  --min-unused-hi-bits-percent <P>   Unused-high-bits threshold (default 70.0)
  --jvm-flags-file <PATH>           External JVM flags file
  --xdebug-max-obj-graph-scan-threads <N>
                                    Object-graph scan threads (0 = auto)
```

Run `ss-mat --help` for the full list.

## Running tests

```bash
cargo test
```

17 unit tests cover array/field size calculations, detection helpers, BFS
array sizing, and HPROF heap-info parsing (`crates/traversal/src/linear_scan.rs`,
`crates/traversal/src/heap_traverser.rs`, `crates/hprof-parser/src/heap_info.rs`).

## Regression check

`scripts/parity.py` extracts the key report metrics and compares them with the
Java reference values in `scripts/parity_baseline.json`:

```bash
./target/release/ss-mat task_dump.hprof /tmp/report.html
python3 scripts/parity.py /tmp/report.html
```

The baseline uses the Java report from the same dump. Sections 8/11/12/18 are
now within tolerance; the remaining expected failures are the live/garbage
split (static roots), Sections 9/10/14, and the Section 13/16/17 detection
stubs.

## Known gaps

- Section 3's "Live shallow" column is live shallow size; dominator-based
  retained size is not implemented yet.
- Static-field roots (`JAVA_STATIC`) are display-only, so live/garbage differs
  from Java by about 1% of objects.
- Section 13/16/17 are heuristic or stub implementations, not full duplicate /
  WeakHashMap detection.
- Section 14 still differs from the Java reference. Section 9 now matches
  Java's container accounting (11,399KB vs 11,110KB, +2.6%), and Section 10
  matches Java's object-array accounting (4,927KB vs 4,959KB, -0.65%) after
  excluding container backing arrays and including garbage arrays.
- Section 14's class-wide chain panel was removed because it extrapolated
  top-by-size samples into impossible sizes (e.g. 12.3 GB for byte[]).
- Section 4's garbage row uses total shallow bytes; Java displays a sampled
  retained tree.

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

## Error handling

- Invalid/truncated HPROF files produce a clear message (not a raw backtrace)
- Non-existent, compressed, or too-small files are rejected up front
- Failures exit with a non-zero status code
