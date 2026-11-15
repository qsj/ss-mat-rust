# Memory acceptance harness

Independent acceptance for the SS-Mat memory optimisation effort. The harness touches
no product code: it builds the workspace, runs the analyser under instrumentation, and
judges three contract items -- output invariance, the memory budget, and growth.

Everything below was measured on the current tree at 2026-11-15T10:10:17+08:00.

| piece | role |
|---|---|
| `scripts/mem_accept.sh` | the acceptance run: build, measure, parity, tests, growth, one PASS/FAIL line per item |
| `scripts/synth_hprof.py` | writes a valid HPROF 1.0.2 dump with a chosen object count, so growth is measured at two sizes |
| `scripts/parity.py` | pre-existing semantic gate (17 metrics, all tolerances 0) |

## 1. Contract and thresholds

The thresholds are named constants at the top of `scripts/mem_accept.sh`; the
table repeats them so the numbers have one home in prose.

| item | measurement | pass condition | constant |
|---|---|---|---|
| C1 output invariance | `python3 scripts/parity.py <report> --strict` | 17/17 metrics within tolerance | `PARITY_EXPECTED_METRICS=17` |
| C1 output invariance | `cargo test --workspace` | every test target compiles and passes | -- |
| C2 budget | `/usr/bin/time -v` "Maximum resident set size" on `task_dump.hprof` | <= 300,000 KB | `MAX_RSS_KB=300000` |
| C2 budget | the binary's own `[mem] peak: ... live_heap=` line | <= 200,000 KB | `MAX_PEAK_LIVE_KB=200000` |
| C3 growth | marginal slope of peak live heap between two synthetic dumps | <= 40 MB per million objects | `GROWTH_MAX_SLOPE_MB_PER_MILLION=40` |
| C3 growth | peak RSS delta between the same two dumps | <= 60 MB | `GROWTH_MAX_RSS_DELTA_MB=60` |

Synthetic sizes for C3 are `GROWTH_SMALL_OBJECTS=1000000` and
`GROWTH_LARGE_OBJECTS=4000000`.

Slope definition: `(peak_live(large) - peak_live(small)) / 1024 / ((objects(large) - objects(small)) / 1e6)`.
A constant-per-object implementation shows one flat MB-per-million figure at both
sizes and a slope equal to that figure; a sub-linear implementation shows a smaller
MB-per-million at the large size and a slope below it.

## 2. How to run

```bash
export PATH=/home/secsmart/localtc/bin:$HOME/.cargo/bin:$PATH
export CC=/home/secsmart/localtc/bin/cc
export CARGO_TARGET_DIR=/home/secsmart/target-harness    # own target dir, never the shared one

cd /home/secsmart/test_workspace/ss-mat-rust
scripts/mem_accept.sh                                     # full acceptance, exit 1 on any failure
scripts/mem_accept.sh --no-growth --no-tests              # budget only, ~90 s
scripts/mem_accept.sh task_dump.hprof --report /tmp/out.html
scripts/mem_accept.sh --dump-dir /tmp/ss-harness --keep   # reuse synthetic dumps, keep the temp dir
```

The harness honours `CARGO_TARGET_DIR` when it is set and reads the binary from
`$CARGO_TARGET_DIR/release/ss-mat`. It prepends `$HOME/.cargo/bin` and
`$HOME/localtc/bin` to `PATH` and defaults `CC` to
`$HOME/localtc/bin/cc` when that file exists. It creates exactly one temp
directory under `/tmp` and removes only that directory on exit (`--keep` keeps it).

Each contract check can also be reproduced by hand:

```bash
BIN=$CARGO_TARGET_DIR/release/ss-mat

# C1a: parity on the reference dump
$BIN task_dump.hprof /tmp/out.html
python3 scripts/parity.py /tmp/out.html --strict

# C1b: tests
cargo test --workspace

# C2: budget on the reference dump
SS_MAT_MEM_TRACE=1 /usr/bin/time -v $BIN task_dump.hprof /tmp/out.html
#   read "Maximum resident set size (kbytes)" from stderr, and the final
#   "[mem] peak: rss=... live_heap=..." line

# C3: growth
python3 scripts/synth_hprof.py --objects 1000000 --out /tmp/synth-1m.hprof
python3 scripts/synth_hprof.py --objects 4000000 --out /tmp/synth-4m.hprof
SS_MAT_MEM_TRACE=1 /usr/bin/time -v $BIN /tmp/synth-1m.hprof /tmp/s1.html
SS_MAT_MEM_TRACE=1 /usr/bin/time -v $BIN /tmp/synth-4m.hprof /tmp/s4.html
```

## 3. Current verdict

```
=== verdict ===
C1 output invariance ............ FAIL
C2 memory budget ................ FAIL
C3 sub-linear growth ............ FAIL

REJECTED: 3 contract item(s) failed.
```

| item | measurement | threshold | current | verdict |
|---|---|---|---|---|
| C1a | `parity.py --strict` on the reference report | 17/17, tolerance 0 | 17/17 | PASS |
| C1b | `cargo test --workspace` | green | `ss-traversal` unit test does not compile: `error[E0560]: struct ss_stats::traversal_data::ObjectEntry has no field named file_offset` (`crates/traversal/src/heap_traverser.rs`) | FAIL |
| C2a | max RSS, reference dump | <= 300,000 KB | 715,984 KB | FAIL |
| C2b | peak live heap, reference dump | <= 200,000 KB | 439,969 KB | FAIL |
| C3a | marginal slope of peak live heap | <= 40 MB per million objects | 105.6 MB per million objects | FAIL |
| C3b | peak RSS delta, 1M -> 4M objects | <= 60 MB | 341.4 MB | FAIL |

Reference dump (267.8 MiB, 2,794,680 objects): wall 82.6 s, max RSS 715,984 KB,
sampled peak RSS 716,412 KB, peak live heap 439,969 KB.

Growth measurement:

| objects | dump size | wall | max RSS | peak live heap | MB per million objects |
|---|---|---|---|---|---|
| 1,000,007 | 36.2 MiB | 1.03 s | 121,336 KB | 108,444 KB | 105.9 |
| 4,000,007 | 145.0 MiB | 4.61 s | 470,932 KB | 432,951 KB | 105.7 |

A flat MB-per-million figure at both sizes is the signature of a per-object cost:
growth is linear, not sub-linear.

### Attribution from the `[mem]` phase lines (reference dump)

Live heap at each phase boundary; the delta column is what that phase added.

| phase line | live heap | delta |
|---|---|---|
| start | 0 KB | -- |
| phase1-parsed | 88,426 KB | +88,426 KB |
| phase3-linear-scan | 135,219 KB | +46,793 KB |
| bfs-object-table | 278,914 KB | +143,695 KB |
| bfs-outgoing-refs | 356,472 KB | +77,558 KB |
| bfs-traversal-done | 393,057 KB | +36,585 KB |
| chain-descriptions | 423,956 KB | +24,399 KB |
| phase4b-humongous (peak live 439,969 KB) | 418,135 KB | -5,821 KB |
| phase4-report / end | 336,256 KB | -81,879 KB |

Largest three consumers, in order: the BFS object table (+140 MB), the Phase 1 parse
indexes (+86 MB), and BFS outgoing-edge extraction (+76 MB). RSS peaks at 716,412 KB
while anonymous memory is 438,720 KB; the remainder is the page-resident mmap of the
268 MiB dump.

On the 4M synthetic dump the same three phases dominate (`bfs-object-table` +203,127 KB,
`bfs-traversal-done` +51,724 KB, `bfs-outgoing-refs` +48,393 KB), i.e. the growth is
the per-object structures the BFS builds, not the parser.

## 4. What the harness actually does

1. `cargo build --release` into `CARGO_TARGET_DIR` (a compile failure stops the run with
   exit 2 -- no contract can be judged against a tree that does not build).
2. Runs the analyser once on the dump with `SS_MAT_MEM_TRACE=1` under
   `/usr/bin/time -v`, redirecting both streams into one log, then parses:
   `Maximum resident set size (kbytes)`, the final `[mem] peak:` line
   (`rss=` and `live_heap=`), and the wall time.
3. Judges C2 from those two numbers and, when it fails, prints the three largest
   live-heap deltas among the `[mem]` phase lines.
4. Judges C1: `parity.py --strict` on the report that step 2 produced, then
   `cargo test --workspace` (`--no-tests` skips the latter).
5. Judges C3: generates or reuses the two synthetic dumps, runs each under the same
   instrumentation, computes MB-per-million for both sizes, the marginal slope and
   the peak-RSS delta (`--no-growth` skips this step).
6. Prints one PASS/FAIL line per contract item and exits 1 if any item failed.

## 5. The synthetic dumps

`scripts/synth_hprof.py --objects N --out PATH` writes a valid HPROF 1.0.2 file. The
layout is dictated by the two readers that must both accept it:
`crates/hprof-parser/src/record_reader.rs` (the strict raw record walker, which
requires every top-level record to be consumed exactly) and the `jvm-hprof` 0.1 crate
(which panics on an unknown top-level tag and asserts the ClassDump constant-pool
length is 0).

File order:

1. header `JAVA PROFILE 1.0.2\0` + `u4 id_size=8` + `u8 timestamp`;
2. nine UTF8 records (0x01) for class names and field names;
3. one LOAD_CLASS record (0x02) per class: `java.lang.Object`, `demo.Node`, `byte[]`,
   `int[]`, `demo.Node[]`;
4. HEAP_DUMP_SEGMENT (0x1C) records: five CLASS_DUMPs (0x20), five ROOT_STICKY_CLASS
   (0x05) pointing at them, one ROOT_JNI_GLOBAL (0x01), one ROOT_THREAD_BLOCK (0x06),
   six PRIMITIVE_ARRAY_DUMPs (0x23: four `byte[]`, two `int[]`), one
   OBJECT_ARRAY_DUMP (0x22) of `demo.Node[]`, then N INSTANCE_DUMPs (0x21);
5. a trailing segment with ROOT_UNKNOWN (0xFF) for each chain head, and a final
   HEAP_DUMP_END (0x2C) record.

`demo.Node` declares exactly the three fields its instance body carries -- a reference,
a byte and an int, body length `8 + 1 + 4 = 13` -- so the analyser's field walker
finds a body whose length matches the field descriptors. The reference field chains
runs of 1000 objects, so every instance is reachable from a chain head and the BFS
sees a live graph instead of one giant garbage pile.

Self-validation: the generator's dumps parse and analyse cleanly, for example

```
Parsed: 5 classes, 4000000 objects, 6 prim arrays, 1 obj arrays, 4007 roots
  Object directory: 4000012 records (4000000 instances, 6 prim arrays, 1 obj arrays, 5 classes), walk 0.18s
```

with exit status 0 and a report written, at every size used by the harness.

Sizes: 36 MiB at 1M objects, 145 MiB at 4M. The harness writes them into its own
temp directory (or `--dump-dir`) and reuses an existing file of the right name instead
of regenerating it; nothing outside that directory is touched.

## 6. Caveats

* Max RSS is load-dependent. Two runs of the same binary on the reference dump
  reported 715,984 KB and 753,124 KB; the allocator-accounted live heap moved far
  less (439,969 KB / 468,898 KB). Judge C2 against the threshold, not against a
  previous run's number.
* The parity baseline in `scripts/parity_baseline.json` belongs to
  `task_dump.hprof`. Running the harness on another dump will fail C1a by
  construction; that is the expected behaviour, not a regression.
* C1b covers the whole workspace, including crates under active edit. A transient
  compile error in a test target is reported as a C1 failure and should be re-run
  before it is read as a regression.
* The generator only emits the record kinds the harness needs to exercise parsing,
  indexing, BFS, primitive arrays and object arrays. It deliberately contains no
  String instances and no stack traces, so those detectors stay idle on synthetic
  input; parity and output invariance are always judged on the reference dump.
