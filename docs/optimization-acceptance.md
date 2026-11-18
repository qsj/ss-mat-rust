# Independent optimization acceptance protocol

## Scope and immutable gates

Baseline is commit `86c5bdb937d890a28b0207a5eb3578f7c850d8e1`, built from
`git archive`, not the concurrently edited working tree. This phase captures only
task/appauditd baselines and prepares review; it does **not** accept an optimized
binary or authorize the 8.6GB run. No product crates are modified.

Acceptance requires all of:

- Task report: `python3 scripts/parity.py REPORT --strict`, exactly 17/17.
- Appauditd debug: `python3 scripts/java_parity.py DEBUG testdata/appauditd/java`,
  all four reference TSVs identical as multisets, including duplicate counts.
- Old/new full forensic bundles and significant report content equivalent; preserve
  Java quirks, exact comparisons, traversal and sample order. No top-row-only proof.
- Final executable on `/home/secsmart/dumps/536803-live.hprof`: GNU time maximum
  RSS **strictly below 1,000,000 KiB**, successful exit. No threshold changes.
- Final workspace tests pass. Freeze source and executable before final measurement;
  any later product edit/build invalidates final evidence and requires rerunning.

Legacy `mem_accept.sh` separately requires task RSS <=300,000 KB, allocator live
heap <=200,000 KB, synthetic heap slope <=40 MB/million and RSS delta <=120 MB.
These are distinct constraints, not replacements for the big-dump limit. Historical
platform exceptions in memory-acceptance.md do not waive this run's requirements.
A fixed base plus a linear object cost can produce falling bytes/object: small
synthetic tests do not establish constant memory or asymptotic sublinear growth.

## Captured oracle

Evidence root: `/home/secsmart/runs/accept-independent-qxmgbH`.
Executable: `bin/ss-mat-baseline` beneath that root. SHA256:
`b3a0c8eb2dbe4720bea08cfb548a303e4d148f6100aecf282ab7adcf542ecb46`.
Source archive checkout: `source/`; identity: `source-commit.txt`; build log/status:
`build.log`, `build.exit`. Isolated target: `/home/secsmart/target-independent-accept`.
Each `{task,appauditd}-{normal,debug}-{1,2}/` holds the report, stdout/stderr,
`time.txt` and `run.exit`; debug runs hold `debug/`. Reference checks retain
`parity.log`/`.exit` and `java-parity.log`/`.exit` where applicable.
Logs/reports can contain sensitive dump material: keep local; print only hashes,
counts and verdicts, never content samples.

## Baseline results and unresolved control differences

All eight analyzer runs exited 0. All four task strict checks passed 17/17,
and all four debug Java checks (task and appauditd, two repeats each) passed
4/4, exit 0. Isolated baseline workspace tests passed 35 tests, exit 0. Checker
mutation tests passed, exit 0. No 8.6GB analyzer run was performed.

| scenario | repeat 1 max RSS KiB | repeat 2 max RSS KiB |
|---|---:|---:|
| task normal | 227496 | 228412 |
| task debug | 238068 | 238148 |
| appauditd normal | 433872 | 440780 |
| appauditd debug | 468608 | 468620 |

All four old/old full comparisons deliberately return exit 1: HTML differs in
multiple sections; both debug controls additionally differ in `ref_chains.tsv`
and `s14_summary.txt`. S14 differs only on line 2, the `generated` field. Chain
differences are **not merely row ordering**: task has 31 unmatched lines on each
side; appauditd has 48/47 unmatched physical lines, corrected below to 47/47
logical records. Remaining 13 debug files, including the entire emitted S8 groups
file, compare equal under the stated policies. This does not mean all original
S8 content is present in the baseline. These baseline
instabilities remain review obligations, not waived equivalence gates. Details:
`*-control.json`, `*-control.exit`, `control-diagnosis.json` in the evidence root.
A column-level exploratory diagnostic exited 1 (variable TSV row widths); no
acceptance result relies on that diagnostic. Existing ref-chain output is kept
byte-exact rather than parsed with an assumed uniform schema.

## Read-only attribution and baseline incompleteness

All source citations in this subsection refer to the isolated baseline commit,
not line numbers in the concurrently edited tree.

**Proven timestamp source:** `debug_dump.rs:290` writes generated using
`timestamp()`; `debug_dump.rs:355-358` reads `SystemTime::now()`. Both
old/old S14 summaries have exactly one generated field and all other bytes equal.
The diagnostic labels this explicitly; the strict comparison still fails.

**Proven chain selection variability:** serializer `debug_dump.rs:394-452`
writes class and humongous records with different column counts, unescaped sample
text (which can contain newlines), and sorts complete records. Logical-record
parsing yields task 467/467 records and appauditd 508/508. Keyed by section,
class, kind and complete path, task has 436 common records and appauditd 461;
every common record is byte-identical in size, count, ordered object IDs and full
sample description. Unmatched records are 31/31 and 47/47 respectively, not
encoded-string-ID substitutions or mere row ordering.

Every replacement is at the minimum retained size in a full ten-group bucket:
24 affected task class/kind buckets and 26 appauditd buckets. Baseline
`heap_traverser.rs:2973-2981` groups with a HashMap and iterates it;
`:3049-3056` sorts only by size then truncates each kind to ten. Thus tied
boundary selection is observable baseline variability with a matching source
mechanism. Total selected size and group counts remain stable by bucket. Selected
object count is stable for appauditd but task class29169/full changes 120 to123
at a 272-byte cutoff. This is not proof all future old/new sample changes are
harmless: preserve algorithm, inputs and ordering; unmatched groups remain
explicitly unresolved, not automatically waived. Detailed payload text is never
printed. Evidence: `control-attribution.json` and `cutoff-attribution.json`.

**Baseline S8 is itself incomplete:** `linear_scan.rs:2152` selects top200;
`:2156-2170` decodes only those groups and stores empty content/sample IDs for
others even in debug mode. `duplicate_strings.rs:49-65` records a clone;
`debug_dump.rs:1134-1139,1183-1196` sorts and serializes these stored values.
Task: 43,441 emitted groups, 43,241 with both zero representative and missing
sample IDs. Appauditd: 97,222 groups, 97,022 with that signature. Empty content
length counts are one higher for each dump because an actual empty-string group
also exists. The checker compares every emitted field, not just top rows, and
reports this baseline limitation separately. Coordinator confirmed preservation
scope: retain existing top200/blank behavior; restoration is separate functionality
and is not authorized here. Newly missing selected content or changed lower-row
fields remain strict failures. Evidence: `s8-completeness.json`.

**Streaming consumer review:** at inspection the new `DupPrimMemberStore`
preserved producer order, duplicates and flags across repeated visits, but was not
yet wired to `DupPrimGroupDebug.members`. Baseline writer `debug_dump.rs:168`
clones groups; `:191-215` computes liveness/flags for every member;
`:225-253` writes group totals/min representative; `:265-266` sorts member
rows by object ID. Integration must preserve these consumer observations and
propagate I/O failures, not silently replace all-members output with samples.
This is an interface obligation, not a verdict on an unfinished implementation.

The approved extension emits a separate diagnostic section: logical records,
common-key exact checks, bucket group/size/count aggregates, hashes of every
unmatched path key with size/cutoff evidence, timestamp and S8 completeness.
No diagnostic changes the strict result or exit status. The same checker
carries an HTML diagnostic that decodes encoded string ids and localizes every
differing section; see the next section. Unescaped chain payloads
can in principle mimic framing markers; ambiguous evidence must be reviewed,
not accepted through the diagnostic parser.

## Comparing complete evidence

Run the new checker against matching scenario/mode bundles:

```bash
python3 scripts/verify_optimization.py BASELINE_BUNDLE CANDIDATE_BUNDLE \
  --debug --control BASELINE_REPEAT_BUNDLE > comparison.json
rc=$?
printf "%s\n" "$rc" > comparison.exit
exit "$rc"
```

Omit `--debug` only for normal-mode runs. Exit 0 means exact equivalence; 1 is a
difference needing review; 2 is invalid/missing required evidence. Compare both
normal and debug modes. The checker requires all four gate TSVs plus S8 groups,
compares the union of **all** debug files (missing files cannot disappear through
an intersection), and compares non-TSV summaries byte-for-byte. Aggregate TSVs
use exact line multisets, preserving multiplicity, full field values, line endings,
S8 content hash/length/preview, group indices and sample-id order within each row.
`objects.tsv`, `roots.tsv`, and `ref_chains.tsv` are compared byte-for-byte to avoid
hiding traversal/sample sequence changes. Full HTML, including hidden details,
scripts and attributes, is byte-compared and differences localized by section.
This intentionally over-reports harmless HTML ordering differences rather than
silently discarding potentially significant output.

Old/old controls are diagnostic, never automatic waivers. For each differing
section/file: retain control comparisons, identify the precise unstable field or
row order, check old/new counts and complete payloads, and review the implementation
for preserved stable tie-breaking/traversal/sample order. Escalate unresolved
differences to the coordinator; do not accept merely because the section also
changed old/old. No generic section exclusion list. Report fingerprinting is a
secondary diagnostic only: existing report_fingerprint.py strips scripts, flattens
section tokens and loses ordering; it cannot prove hidden payload equivalence.
Existing java_parity.py --all intersects file names and is not a missing-file gate.
The four standard Java gates alone do not detect missing S8 group content.

Mutation tests live in the evidence root as `test_checker.py`: multiset reorder,
duplicate loss, hidden S8 script loss, chain order, missing extra files, required
S8 file, exit 0/1/2 and no old/old waiver. Run with PYTHONDONTWRITEBYTECODE=1 to
avoid writing caches into the shared repository.

## HTML equivalence: decoded-identifier diagnosis

The four retained old/new pairs still fail the strict HTML byte comparison:
`old-new-*-strict.exit` and `old-new-*-html.exit` in the HTML evidence root all
contain `1`. This subsection enumerates every differing section and attributes
it. It waives nothing: the strict comparator, its equality and its exit status
are unchanged, and byte-identical HTML is **not** claimed.

Method. `scripts/verify_optimization.py --html-diagnostic` reads each report's
own `ENCODED_STRINGS` table, replaces every `|*/c<idx>|`/`|*/s<idx>|`
placeholder with the tag the client-side decoder would insert, and only then
slices the `id="Section-N"` sections and compares them. Each section is split
into its table skeleton and its top-level `<details>` payload blocks; payload
rows are compared as exact multisets and payload identities as summary hashes,
so "same payloads, different order" is reported separately from "different
payload set". Output holds counts, hashes and section keys only, never report
text. The flag adds an `html_diagnostic` object; it never changes `comparison`,
`equal` or the process exit code.

Results for the four old/new pairs (task/appauditd x normal/debug), all exit 1:

- No unresolved placeholder remains (`unresolved_placeholders` is `[0, 0]`).
- Sections that differ only in placeholder indices, identical after decoding:
  `6:7`, `9:10`, `10:11`, `13:14`, `16:17`, plus `11:12` and `17:18` in the
  pairs where those two later compare equal. Identifier assignment is not
  content.
- The preamble's head (styles, JS decoder, `<header>`, advisor summary) and
  its tail are byte-identical; only the `ENCODED_STRINGS` table differs, which
  is the identifier consequence of the payload changes below.
- Every other differing section is listed in the table. "Payload blocks" counts
  top-level `<details>` payloads that differ, out of the blocks in the section;
  "skeleton" counts entry rows outside `<details>`.
  "baseline/self" and "candidate/self" count same-binary repeat pairs, of 16
  baseline pairs (four baseline runs per dump plus the four retained old/old
  controls) and 9 candidate pairs (reruns of the frozen executable, normal and
  debug), in which the section also differs after decoding.

| section | content | old/new pairs | payload blocks | skeleton | mechanism | class | baseline/self | candidate/self |
|---|---|---|---:|---|---:|---:|---:|---:|
| preamble | head, JS decoder, header, advisor | 4/4 | encoded table only | none | M1/M2 | (b)/(a) | 16/16 | 9/9 |
| 1:2 | S2 OOM thread local variables | 2/4 | 5 of 6 | none | M3 | (a) | 6/16 | 3/9 |
| 2:3 | S3 by class: chain panels | 4/4 | 36-48 of 64/65 | none | M1 | (b) | 16/16 | 9/9 |
| 3:4 | S4 by GC root | 4/4 | 89-91 of 101 | 8-18 rows | M2 | (a) | 16/16 | 9/9 |
| 7:8 | S8 duplicate strings panel | 4/4 | 1 of 4 | none | M1 | (b) | 16/16 | 9/9 |
| 8:9 | S9 bad collections | 4/4 | 1 of 4 | 0-2 rows | M1 + M4 | (b)+(a) | 15/16 | 9/9 |
| 11:12 | S12 boxed numbers panel | 2/4 | 1 of 3 | none | M1 | (b) | 8/16 | 4/9 |
| 12:13 | S13 duplicate objects panel | 4/4 | 1 of 3 | none | M1 | (b) | 15/16 | 9/9 |
| 15:16 | S16 duplicate lists panel | 4/4 | 1 of 3 | none | M1 | (b) | 15/16 | 9/9 |
| 17:18 | S18 DirectByteBuffer per-type | 2/4 | 1 of 4 | none | M4 | (a) | 8/16 | 6/9 |
| 18:19 | S19 humongous chain panels | 1/4 | 2 of 2 | none | M1 | (b) | 4/16 | 1/9 |

Sections `0:1`, `4:5`, `5:6`, `14:15`, `19:20`, `20:21`, `21:22`, `22:23` and
`23:24` are byte-identical in all four pairs after decoding.
The preamble row is not a content section: only the identifier table differs,
and its strings are exactly the decoded payloads of M1/M2, so it is attributed
to those mechanisms rather than to a separate cause.

**M1 - tied ten-group reference-chain cutoff (previously documented).**
`crates/traversal/src/heap_traverser.rs:2973` collects chain groups in a
`HashMap<Vec<(u32, u32)>, Vec<u32>>` and iterates it; `:3051-3056` sorts each
kind by `size_bytes` alone and truncates to ten. `sort_by` is stable, so groups
whose sizes tie keep HashMap iteration order, and which tied groups survive the
truncation is run-dependent. `:3145` and `:3220-3223` are the same pattern for
the humongous store. The HTML panels render that store
(`crates/report/src/main_report.rs:173`,
`crates/report/src/ref_chain_render.rs:377,412`), and Section 3 also walks the
group list to pick its three per-instance samples
(`crates/report/src/main_report.rs:809-811`). Same-bundle chain evidence: every
unmatched bucket in the old/new and old/old debug comparisons is at the
ten-group cutoff (`both_at_ten_group_cutoff`: 24/24 task and 26/26 appauditd
unmatched buckets), every common record is byte-identical (task 467/467
logical records with 432 common keys; appauditd 508/508 with 455; the
common-key count varies with the pair compared because the selection itself is
the varying quantity), and the bucket aggregates of group count, total size and
total count are stable on the old/new pairs. All cited line ranges are identical in the baseline commit.

**M2 - tied GC-root selection (Section 4).**
`heap_traverser.rs:4447` groups roots in a `HashMap<(RootType, u32), ...>` and
iterates it; `:4490` sorts only by retained bytes; `:4493` keeps
`GC_ROOT_TOP_N = 100` (`:4364`); `:4484` picks the representative with
`max_by_key`, which returns the last maximum in HashMap order. Display-only
`Java Static [...]` roots are inserted with retained bytes and count 0
(`:4470-4473`), so thousands of them tie at the bottom of the sort and the cut
keeps a run-dependent 100: 77-80 of the 101 payload summaries differ per pair,
89-91 payload trees differ and 8-18 entry rows differ, in every compared
pair (4 old/new, 20 same-binary). Totals outside the tie stay equal.

**M3 - OOM local variables from a nondeterministically chosen equal-depth thread.**
`crates/cli/src/main.rs:145` iterates `heap_arc.stack_traces` (a `HashMap`) and
keeps the first thread it encounters among equal `frames.len()` maxima, while
the report independently re-selects the displayed thread from a sorted serial
list
(`crates/report/src/main_report.rs:651`). Threads with equal depth have the same
frames but different local-variable objects, so when the two selections differ
the displayed frames are unchanged while the local-variable payload values
differ. Two stable states are observed and both binaries produce both of them.

**M4 - single-key sort or no sort over HashMap iteration in table rows.**
Section 9's per-class table sorts `entries` by wasted bytes alone
(`crates/report/src/main_report.rs:1323`, baseline `:1317`), so classes tied at
the same wasted-byte value swap and their alternating row classes swap with
them (0-2 skeleton rows). Section 18's "Per-type breakdown" renders
`report.per_class` in HashMap iteration order without any sort
(`crates/traversal/src/linear_scan.rs:4970`, baseline `:4806`), so its entries
can swap, and `per_class.first()` also names the reference-chain panel that
follows.

Controls. Same-binary repetitions use the frozen baseline
(`bin/ss-mat-baseline`, SHA256 `b3a0c8eb...`) and the frozen candidate
(`a03f66bb...`, `/home/secsmart/target-bounded-scan-s10/release/ss-mat`).
Reports, debug directories, exit codes and timings live under
`/home/secsmart/runs/accept-html-2/{selfcontrol,oidcontrol,repeats}`; the
classifier output is `classification.json` / `classification.md` there. The
candidate fixpoint is not stable: 1 of 7 candidate runs and 1 of 4 baseline
runs render the other Section 19 state, which is why only the appauditd debug
old/new pair shows `18:19`.

Unresolved list:

- No section is classified (c). Every differing section is either identical
  input data whose ordering or selection depends on HashMap iteration (a), or
  the previously documented tied ten-group cutoff (b); no unexplained content
  difference remains.
- Byte-identical HTML remains unresolved and is not claimed. Both binaries vary
  run-to-run in the sections above, so no single candidate run can be expected
  to match a single baseline run byte-for-byte; the strict gate stays exit 1.
- The two documented non-HTML residuals are unchanged and not waived:
  `ref_chains.tsv` tied-cutoff selection and the `s14_summary.txt` `generated`
  timestamp.

Reproduction (all commands exit 1 on the retained bundles; the diagnostic
object is the evidence, not a verdict):

```bash
export PYTHONDONTWRITEBYTECODE=1 PATH=/home/secsmart/localtc/bin:$HOME/.cargo/bin:$PATH
B=/home/secsmart/runs/accept-independent-qxmgbH
C=/home/secsmart/runs/bounded-flush-WHTNiI
O=/home/secsmart/runs/accept-html-2
for mode in normal debug; do
  for scen in task appauditd; do
    case $scen in task) cdir="$C/task-$mode";; *) cdir="$C/app-$mode";; esac
    if [ "$mode" = debug ]; then extra=--debug; else extra=; fi
    python3 scripts/verify_optimization.py "$B/$scen-$mode-1" "$cdir" $extra \
      --html-diagnostic --control "$B/$scen-$mode-2" \
      > "$O/old-new-$scen-$mode.html.json"
    printf '%s\n' "$?" > "$O/old-new-$scen-$mode.html.exit"
  done
done
python3 "$O/classify.py"   # regenerate classification.json/.md from the pairs
```

Same-binary controls are produced by running the frozen executables twice per
dump, then comparing the two bundle roots with the same command and no
`--control`; `repeats/` holds the pairwise summaries. Checker mutation tests
still pass (`test_checker.py`, exit 0) and `--html-diagnostic` does not alter
any strict result.

## Final independent measurement (coordinator schedules first)

Do not run this concurrently with another large-dump analysis. Use an absolute
path to a pinned copy of the **final** binary; record source commit plus working
tree patch identity, build command/log, toolchain versions and binary SHA256.
Record dump identity and scenario flags. Do not invoke `cargo run` or rebuild
between measurement and acceptance. Example shell fragment after scheduling:

```bash
set -uo pipefail
export PATH=/home/secsmart/localtc/bin:$HOME/.cargo/bin:$PATH
export CC=/home/secsmart/localtc/bin/cc
export CARGO_TARGET_DIR=/home/secsmart/target-independent-accept
# Set these to existing pinned binary and a NEW owned output directory.
: "${FINAL_BIN:?absolute final executable path required}"
: "${OUT:?new owned evidence directory required}"
case "$FINAL_BIN" in /*) ;; *) exit 2;; esac
mkdir "$OUT" || exit 2
sha256sum "$FINAL_BIN" > "$OUT/binary.before.sha256" || exit 2
unset SS_MAT_DEBUG_DIR
LC_ALL=C SS_MAT_MEM_TRACE=1 /usr/bin/time -v -o "$OUT/time.txt" \
  "$FINAL_BIN" /home/secsmart/dumps/536803-live.hprof "$OUT/report.html" \
  > "$OUT/stdout.log" 2> "$OUT/stderr.log"
rc=$?
printf "%s\n" "$rc" > "$OUT/run.exit"
sha256sum "$FINAL_BIN" > "$OUT/binary.after.sha256" || exit 2
cmp "$OUT/binary.before.sha256" "$OUT/binary.after.sha256" || exit 2
[ "$rc" -eq 0 ] || exit "$rc"
python3 - "$OUT/time.txt" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
values = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
assert len(values) == 1, "missing or ambiguous max RSS"
rss = int(values[0])
print("max_RSS_KiB", rss, "strict_limit", 1000000)
sys.exit(0 if 0 < rss < 1000000 else 1)
PY
exit $?
```

GNU time maximum RSS is authoritative here, not phase-sampled RSS and not allocator
heap. Record debug-mode resource results separately; require debug equivalence
even when normal mode is the final RSS scenario. Retain elapsed/user/system times,
page faults and environment/load/cache state. Do not claim no time regression
without matched baseline/final environment measurements. Complete output and
reference comparisons remain mandatory even when the RSS limit passes.
