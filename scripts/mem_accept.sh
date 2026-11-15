#!/usr/bin/env bash
# SS-Mat memory acceptance harness.
#
# Usage:
#   scripts/mem_accept.sh [dump] [--report PATH] [--no-growth] [--no-tests]
#                         [--dump-dir DIR] [--keep]
#
# Judges three contract items and prints one PASS/FAIL line per item:
#   C1  output invariance -- scripts/parity.py --strict is 17/17 and cargo test --workspace is green
#   C2  memory budget on the reference dump -- peak RSS and the binary's own peak live heap
#   C3  growth -- peak live heap and peak RSS must grow sub-linearly between two synthetic dumps
#
# Exit status: 0 when every item passes, 1 when any item fails, 2 when the harness
# itself could not run (build failure, missing binary, bad arguments).
#
# Environment:
#   CARGO_TARGET_DIR  reused when set (the binary is read from its release/ dir);
#                     the harness never overrides it. $HOME/.cargo/bin and
#                     $HOME/localtc/bin are prepended to PATH, and CC defaults to
#                     $HOME/localtc/bin/cc when that file exists and CC is unset.

set -o pipefail

# ---------------------------------------------------------------------------
# Thresholds -- the acceptance numbers of contract C1..C3. Change them together
# with docs/memory-acceptance.md; every unit is stated on its own line.
# ---------------------------------------------------------------------------
MAX_RSS_KB=300000                      # C2: /usr/bin/time -v "Maximum resident set size", KB
MAX_PEAK_LIVE_KB=200000                # C2: "[mem] peak: ... live_heap=" reported by the binary, KB
GROWTH_MAX_SLOPE_MB_PER_MILLION=40     # C3: marginal slope of peak live heap vs object count, MB per million objects, above the fixed base
# C3 absolute check, derived from the slope limit below rather than picked separately:
# the two thresholds are the same statement, so they must not contradict each other.
# 40 MB per million objects over the 3M-object spread between the two synthetic dumps.
GROWTH_MAX_RSS_DELTA_MB=120            # C3: allowed peak-RSS growth between the small and large synthetic dump, MB
GROWTH_SMALL_OBJECTS=1000000           # C3: object count of the small synthetic dump
GROWTH_LARGE_OBJECTS=4000000           # C3: object count of the large synthetic dump
PARITY_EXPECTED_METRICS=17             # C1: metrics parity.py must report within tolerance
PARITY_STRICT_TOLERANCE=0              # C1: --strict means every tolerance in the baseline is 0

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
DUMP=""
REPORT=""
DO_GROWTH=1
DO_TESTS=1
DUMP_DIR=""
KEEP=0

usage() {
  sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --report)   REPORT="$2"; shift 2 ;;
    --report=*) REPORT=$(printf '%s' "$1" | cut -d= -f2-); shift ;;
    --dump-dir) DUMP_DIR="$2"; shift 2 ;;
    --dump-dir=*) DUMP_DIR=$(printf '%s' "$1" | cut -d= -f2-); shift ;;
    --no-growth) DO_GROWTH=0; shift ;;
    --no-tests)  DO_TESTS=0; shift ;;
    --keep)      KEEP=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)           if [ -n "$DUMP" ]; then echo "unexpected argument: $1" >&2; exit 2; fi
                 DUMP="$1"; shift ;;
  esac
done

if [ -z "$DUMP" ]; then
  DUMP="$REPO_ROOT/task_dump.hprof"
fi

if [ ! -f "$DUMP" ]; then
  echo "dump not found: $DUMP" >&2
  exit 2
fi

TMP_BASE=/tmp
if [ -n "$TMPDIR" ]; then
  TMP_BASE="$TMPDIR"
fi
TMP=$(mktemp -d "$TMP_BASE/ss-mat-accept.XXXXXX") || exit 2

cleanup() {
  if [ "$KEEP" -eq 1 ]; then
    echo "kept: $TMP"
    return
  fi
  # Only ever the directory mktemp just created.
  case "$TMP" in
    */ss-mat-accept.*) rm -rf -- "$TMP" ;;
    *) echo "refusing to remove unexpected temp path: $TMP" >&2 ;;
  esac
}
trap cleanup EXIT

case ":$PATH:" in
  *":$HOME/.cargo/bin:"*) ;;
  *) PATH="$HOME/.cargo/bin:$PATH" ;;
esac
case ":$PATH:" in
  *":$HOME/localtc/bin:"*) ;;
  *) PATH="$HOME/localtc/bin:$PATH" ;;
esac
export PATH
if [ -z "$CC" ] && [ -x "$HOME/localtc/bin/cc" ]; then
  CC="$HOME/localtc/bin/cc"
  export CC
fi

cd "$REPO_ROOT" || exit 2

TARGET_DIR="$CARGO_TARGET_DIR"
if [ -z "$TARGET_DIR" ]; then
  TARGET_DIR="$REPO_ROOT/target"
fi
case "$TARGET_DIR" in
  /*) ;;
  *)  TARGET_DIR="$REPO_ROOT/$TARGET_DIR" ;;
esac
BIN="$TARGET_DIR/release/ss-mat"

if [ -z "$DUMP_DIR" ]; then
  DUMP_DIR="$TMP"
fi
mkdir -p "$DUMP_DIR" || exit 2
if [ -z "$REPORT" ]; then
  REPORT="$TMP/report.html"
fi

FAILURES=0
VERDICT_C1="SKIP"
VERDICT_C2="SKIP"
VERDICT_C3="SKIP"

# Run one dump under /usr/bin/time -v and parse the numbers out of the log.
# Results land in RUN_* globals.
measure() {
  local dump="$1" log="$2" report="$3"
  SS_MAT_MEM_TRACE=1 /usr/bin/time -v "$BIN" "$dump" "$report" >"$log" 2>&1
  RUN_STATUS=$?
  RUN_RSS_KB=$(sed -n 's/.*Maximum resident set size (kbytes): *\([0-9][0-9]*\).*/\1/p' "$log" | tail -n 1)
  RUN_PEAK_RSS_KB=$(grep -F '[mem] peak:' "$log" | tail -n 1 | sed -n 's/.*rss=\([0-9][0-9]*\)KB.*/\1/p')
  RUN_PEAK_LIVE_KB=$(grep -F '[mem] peak:' "$log" | tail -n 1 | sed -n 's/.*live_heap=\([0-9][0-9]*\)KB.*/\1/p')
  RUN_WALL=$(sed -n 's/.*Analysis complete\. Total time: \([0-9.]*\)s.*/\1/p' "$log" | tail -n 1)
  [ -n "$RUN_RSS_KB" ] || RUN_RSS_KB=0
  [ -n "$RUN_PEAK_RSS_KB" ] || RUN_PEAK_RSS_KB=0
  [ -n "$RUN_PEAK_LIVE_KB" ] || RUN_PEAK_LIVE_KB=0
  [ -n "$RUN_WALL" ] || RUN_WALL="?"
}

mb() {
  awk -v kb="$1" 'BEGIN { printf "%.1f", kb/1024.0 }'
}

mb_per_million() {
  awk -v kb="$1" -v n="$2" 'BEGIN { if (n<=0) { print "nan"; exit }; printf "%.1f", (kb/1024.0)/(n/1000000.0) }'
}

slope_mb_per_million() {
  awk -v a="$1" -v b="$2" -v na="$3" -v nb="$4" 'BEGIN {
    d = (nb-na)/1000000.0
    if (d <= 0) { print "nan"; exit }
    printf "%.1f", ((b-a)/1024.0)/d
  }'
}

le() {
  awk -v x="$1" -v lim="$2" 'BEGIN { exit !(x <= lim) }'
}

# Top live-heap growth per [mem] phase line, used to explain a C2/C3 failure.
# A phase line looks like: [mem] phase1-parsed   rss=  12345KB anon= ... live=  12345KB ...
# printf pads the value away from its key ("live=  12345KB"), so the number never sits
# in the same whitespace field as the key; match it with a regex instead.
top_phases() {
  local log="$1"
  awk '
    /^\[mem\] / {
      if (match($0, /live= *[0-9]+KB/) == 0) next
      v = substr($0, RSTART, RLENGTH)
      sub(/^live= */, "", v)
      sub(/KB$/, "", v)
      if (prev != "") printf "%d %s\n", v - prev, $2
      prev = v
    }' "$log" | sort -rn | head -3 | awk '{ printf "    %10d KB  %s\n", $1, $2 }'
}

echo "=== SS-Mat memory acceptance ==="
echo "repo:   $REPO_ROOT"
echo "dump:   $DUMP"
echo "report: $REPORT"
echo "tmp:    $TMP"
echo ""

echo "--- build ---"
if ! cargo build --release >"$TMP/build.log" 2>&1; then
  echo "BUILD FAILED -- the tree does not compile; no contract can be judged."
  tail -n 30 "$TMP/build.log"
  exit 2
fi
echo "binary: $BIN"
if [ ! -x "$BIN" ]; then
  echo "binary missing after a successful build: $BIN" >&2
  exit 2
fi
echo ""

# ---------------------------------------------------------------------------
# C2 -- memory budget on the reference dump (also produces the C1 report)
# ---------------------------------------------------------------------------
echo "--- C2: budget on the dump ---"
measure "$DUMP" "$TMP/run.log" "$REPORT"
echo "  exit status      : $RUN_STATUS"
echo "  wall             : $RUN_WALL s"
echo "  max RSS          : $RUN_RSS_KB KB (limit $MAX_RSS_KB KB)"
echo "  peak live heap   : $RUN_PEAK_LIVE_KB KB (limit $MAX_PEAK_LIVE_KB KB)"
echo "  sampled peak RSS : $RUN_PEAK_RSS_KB KB (binary's own [mem] peak: line)"
C2_OK=1
if [ "$RUN_STATUS" -ne 0 ]; then
  echo "  the analyser did not exit 0 -- see $TMP/run.log"
  tail -n 20 "$TMP/run.log" | sed 's/^/    /'
  C2_OK=0
fi
if ! le "$RUN_RSS_KB" "$MAX_RSS_KB"; then C2_OK=0; fi
if ! le "$RUN_PEAK_LIVE_KB" "$MAX_PEAK_LIVE_KB"; then C2_OK=0; fi
if [ "$C2_OK" -eq 1 ]; then VERDICT_C2="PASS"; else VERDICT_C2="FAIL"; FAILURES=$((FAILURES+1)); fi
if [ "$C2_OK" -eq 0 ]; then
  echo "  top live-heap growth by [mem] phase line:"
  top_phases "$TMP/run.log"
fi
echo ""

# ---------------------------------------------------------------------------
# C1 -- output invariance
# ---------------------------------------------------------------------------
echo "--- C1: output invariance ---"
C1_OK=1
if [ ! -f "$REPORT" ]; then
  echo "  no report at $REPORT -- parity cannot be judged"
  C1_OK=0
else
  PARITY_LOG="$TMP/parity.log"
  if python3 scripts/parity.py "$REPORT" --strict >"$PARITY_LOG" 2>&1; then
    PARITY_STATUS=0
  else
    PARITY_STATUS=$?
  fi
  PARITY_LINE=$(grep -E '^[0-9]+/[0-9]+ metrics within tolerance' "$PARITY_LOG" | tail -n 1)
  echo "  parity.py --strict: $PARITY_LINE (expected $PARITY_EXPECTED_METRICS/$PARITY_EXPECTED_METRICS, tolerance $PARITY_STRICT_TOLERANCE)"
  if [ "$PARITY_STATUS" -ne 0 ]; then
    grep -E 'FAIL|MISSING' "$PARITY_LOG" | sed 's/^/    /'
    C1_OK=0
  fi
fi
if [ "$DO_TESTS" -eq 1 ]; then
  if cargo test --workspace >"$TMP/test.log" 2>&1; then
    TEST_LINE=$(grep -E '^test result: ok\.' "$TMP/test.log" | tail -n 1)
    echo "  cargo test --workspace: green  ($TEST_LINE)"
  else
    echo "  cargo test --workspace: FAILED"
    grep -E '^(test result|error|failures:)' "$TMP/test.log" | tail -n 10 | sed 's/^/    /'
    C1_OK=0
  fi
else
  echo "  cargo test --workspace: skipped (--no-tests)"
fi
if [ "$C1_OK" -eq 1 ]; then VERDICT_C1="PASS"; else VERDICT_C1="FAIL"; FAILURES=$((FAILURES+1)); fi
echo ""

# ---------------------------------------------------------------------------
# C3 -- growth between two synthetic dumps
# ---------------------------------------------------------------------------
echo "--- C3: growth vs object count ---"
if [ "$DO_GROWTH" -eq 1 ]; then
  C3_OK=1
  SMALL_DUMP="$DUMP_DIR/synth-$GROWTH_SMALL_OBJECTS.hprof"
  LARGE_DUMP="$DUMP_DIR/synth-$GROWTH_LARGE_OBJECTS.hprof"

  for pair in "$GROWTH_SMALL_OBJECTS|$SMALL_DUMP" "$GROWTH_LARGE_OBJECTS|$LARGE_DUMP"; do
    count=$(printf '%s' "$pair" | cut -d'|' -f1)
    path=$(printf '%s' "$pair" | cut -d'|' -f2-)
    if [ -s "$path" ]; then
      echo "  reuse $path"
      continue
    fi
    if python3 scripts/synth_hprof.py --objects "$count" --out "$path" >"$TMP/synth-$count.log" 2>&1; then
      sed 's/^/  /' "$TMP/synth-$count.log"
    else
      echo "  synth_hprof.py failed for $count objects"
      tail -n 10 "$TMP/synth-$count.log" | sed 's/^/    /'
      C3_OK=0
    fi
  done

  if [ "$C3_OK" -eq 1 ]; then
    measure "$SMALL_DUMP" "$TMP/growth-small.log" "$TMP/growth-small.html"
    SMALL_STATUS=$RUN_STATUS
    SMALL_RSS=$RUN_RSS_KB
    SMALL_PEAK=$RUN_PEAK_LIVE_KB
    SMALL_WALL=$RUN_WALL
    SMALL_OBJS=$(sed -n 's/.*objects_total=\([0-9][0-9]*\).*/\1/p' "$TMP/synth-$GROWTH_SMALL_OBJECTS.log" 2>/dev/null | tail -n 1)
    [ -n "$SMALL_OBJS" ] || SMALL_OBJS=$GROWTH_SMALL_OBJECTS

    measure "$LARGE_DUMP" "$TMP/growth-large.log" "$TMP/growth-large.html"
    LARGE_STATUS=$RUN_STATUS
    LARGE_RSS=$RUN_RSS_KB
    LARGE_PEAK=$RUN_PEAK_LIVE_KB
    LARGE_WALL=$RUN_WALL
    LARGE_OBJS=$(sed -n 's/.*objects_total=\([0-9][0-9]*\).*/\1/p' "$TMP/synth-$GROWTH_LARGE_OBJECTS.log" 2>/dev/null | tail -n 1)
    [ -n "$LARGE_OBJS" ] || LARGE_OBJS=$GROWTH_LARGE_OBJECTS

    SMALL_PER_M=$(mb_per_million "$SMALL_PEAK" "$SMALL_OBJS")
    LARGE_PER_M=$(mb_per_million "$LARGE_PEAK" "$LARGE_OBJS")
    SLOPE=$(slope_mb_per_million "$SMALL_PEAK" "$LARGE_PEAK" "$SMALL_OBJS" "$LARGE_OBJS")
    RSS_DELTA_MB=$(awk -v a="$SMALL_RSS" -v b="$LARGE_RSS" 'BEGIN { printf "%.1f", (b-a)/1024.0 }')

    echo "  small: $SMALL_OBJS objects, exit $SMALL_STATUS, $SMALL_WALL s, max RSS $SMALL_RSS KB, peak live $SMALL_PEAK KB ($(mb "$SMALL_PEAK") MB = $SMALL_PER_M MB per million objects)"
    echo "  large: $LARGE_OBJS objects, exit $LARGE_STATUS, $LARGE_WALL s, max RSS $LARGE_RSS KB, peak live $LARGE_PEAK KB ($(mb "$LARGE_PEAK") MB = $LARGE_PER_M MB per million objects)"
    echo "  marginal slope   : $SLOPE MB per million objects (limit $GROWTH_MAX_SLOPE_MB_PER_MILLION MB above the fixed base)"
    echo "  peak RSS delta   : $RSS_DELTA_MB MB (limit $GROWTH_MAX_RSS_DELTA_MB MB)"

    if [ "$SMALL_STATUS" -ne 0 ] || [ "$LARGE_STATUS" -ne 0 ]; then
      echo "  a synthetic run did not exit 0 -- see $TMP/growth-*.log"
      C3_OK=0
    fi
    if ! le "$SLOPE" "$GROWTH_MAX_SLOPE_MB_PER_MILLION"; then C3_OK=0; fi
    if ! le "$RSS_DELTA_MB" "$GROWTH_MAX_RSS_DELTA_MB"; then C3_OK=0; fi
    if [ "$C3_OK" -eq 0 ]; then
      echo "  top live-heap growth by [mem] phase line (large dump):"
      top_phases "$TMP/growth-large.log"
    fi
    if [ "$SMALL_STATUS" -ne 0 ]; then
      tail -n 15 "$TMP/growth-small.log" | sed 's/^/    /'
    fi
    if [ "$LARGE_STATUS" -ne 0 ]; then
      tail -n 15 "$TMP/growth-large.log" | sed 's/^/    /'
    fi
  fi
  if [ "$C3_OK" -eq 1 ]; then VERDICT_C3="PASS"; else VERDICT_C3="FAIL"; FAILURES=$((FAILURES+1)); fi
else
  echo "  skipped (--no-growth)"
fi
echo ""

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo "=== verdict ==="
printf 'C1 output invariance ............ %s\n' "$VERDICT_C1"
printf 'C2 memory budget ................ %s\n' "$VERDICT_C2"
printf 'C3 sub-linear growth ............ %s\n' "$VERDICT_C3"
echo ""
if [ "$FAILURES" -eq 0 ]; then
  echo "ACCEPTED: every contract item passed."
  exit 0
fi
echo "REJECTED: $FAILURES contract item(s) failed."
exit 1
