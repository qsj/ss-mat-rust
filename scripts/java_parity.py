#!/usr/bin/env python3
"""Compare SS-Mat Rust forensic TSVs with the Java reference fixtures.

Usage:
    python3 scripts/java_parity.py RUST_DEBUG_DIR JAVA_DIR [--all] [--quiet]

RUST_DEBUG_DIR is the directory passed to SS_MAT_DEBUG_DIR; JAVA_DIR is one of
the fixture directories under testdata/ (for example testdata/appauditd/java).

The gate files are compared as multisets of lines, so row order (hash-map
iteration order differs between the two implementations) does not matter, but
duplicate rows are counted. Exit status is 0 only when every gate file matches.

Gate files:
    problem_stats.tsv        S9/S10/S11/S12 per-class problem aggregates
    boxed_summary.tsv        S9/S12 summaries and per-kind rows
    s8_summary.tsv           S8 duplicate-string totals
    s15_dup_obj_arrays.tsv   S15 duplicate-object-array report (Java two-phase)

With --all, every file present in both directories is compared.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

GATE_FILES = [
    "problem_stats.tsv",
    "boxed_summary.tsv",
    "s8_summary.tsv",
    "s15_dup_obj_arrays.tsv",
]


def read_lines(path: Path) -> Counter:
    with path.open("rb") as f:
        return Counter(f.read().splitlines())


def show_diff(name: str, only_rust: Counter, only_java: Counter) -> None:
    print(f"  {name}: FAIL")
    print(f"    only in Rust: {sum(only_rust.values())} line(s)")
    for line, count in list(only_rust.items())[:3]:
        print(f"      +{count} {line[:200]!r}")
    print(f"    only in Java: {sum(only_java.values())} line(s)")
    for line, count in list(only_java.items())[:3]:
        print(f"      -{count} {line[:200]!r}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2

    rust_dir, java_dir = Path(args[0]), Path(args[1])
    if not rust_dir.is_dir() or not java_dir.is_dir():
        print(f"not a directory: {rust_dir} / {java_dir}", file=sys.stderr)
        return 2

    if "--all" in flags:
        names = sorted(
            p.name
            for p in rust_dir.iterdir()
            if p.is_file() and (java_dir / p.name).is_file()
        )
    else:
        names = GATE_FILES

    if not names:
        print("no common TSV files found", file=sys.stderr)
        return 2

    failures = 0
    for name in names:
        rust_path, java_path = rust_dir / name, java_dir / name
        if not rust_path.is_file() or not java_path.is_file():
            print(f"  {name}: MISSING")
            failures += 1
            continue
        rust_lines = read_lines(rust_path)
        java_lines = read_lines(java_path)
        only_rust = rust_lines - java_lines
        only_java = java_lines - rust_lines
        if not only_rust and not only_java:
            if "--quiet" not in flags:
                print(f"  {name}: PASS ({sum(rust_lines.values())} lines)")
        else:
            show_diff(name, only_rust, only_java)
            failures += 1

    if failures:
        print(f"\nFAILED: {failures} of {len(names)} file(s) differ")
        return 1
    print(f"\nPASS: {len(names)} file(s) identical as multisets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
