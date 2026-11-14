#!/usr/bin/env python3
"""Compare key SS-Mat report metrics against the Java reference baseline.

Usage:
    python3 scripts/parity.py /tmp/report.html
    python3 scripts/parity.py /tmp/report.html --baseline scripts/parity_baseline.json --json

The parser is intentionally narrow: it reads the metric lines that SS-Mat and
the Java JXRay report share, so it can be used as a regression gate without
diffing the whole 4MB HTML document.
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from pathlib import Path


def section_text(doc: str, number: int) -> str:
    marker = f'id="Section-{number}"'
    start = doc.find(marker)
    if start < 0:
        return ""
    next_marker = f'id="Section-{number + 1}"'
    end = doc.find(next_marker, start + len(marker))
    if end < 0:
        end = len(doc)
    segment = doc[start:end]
    segment = re.sub(r"<script.*?</script>", " ", segment, flags=re.S | re.I)
    segment = re.sub(r"<style.*?</style>", " ", segment, flags=re.S | re.I)
    segment = re.sub(r"<[^>]+>", " ", segment)
    segment = html_lib.unescape(segment).replace("\xa0", " ")
    return re.sub(r"\s+", " ", segment)


def number(value: str) -> int:
    return int(value.replace(",", ""))


def find_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return number(match.group(1)) if match else None


def extract(doc: str) -> dict[str, int]:
    values: dict[str, int] = {}

    s1 = section_text(doc, 1)
    match = re.search(r"Objects\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", s1)
    if match:
        values["instances"] = number(match.group(1))
        values["object_arrays"] = number(match.group(2))
        values["primitive_arrays"] = number(match.group(3))
        values["total_objects"] = number(match.group(4))
    match = re.search(r"Bytes\s+[\d,]+KB\s+\([\d.]+%\).*?([\d,]+)KB\s+\(100\.0%\)", s1)
    if match:
        values["heap_total_kb"] = number(match.group(1))

    s5 = section_text(doc, 5)
    match = re.search(
        r"Live objects\s+([\d,]+)\s+([\d,]+)KB.*?"
        r"Garbage \(unreachable\) objects\s+([\d,]+)\s+([\d,]+)KB",
        s5,
    )
    if match:
        values["live_objects"] = number(match.group(1))
        values["live_bytes_kb"] = number(match.group(2))
        values["garbage_objects"] = number(match.group(3))
        values["garbage_bytes_kb"] = number(match.group(4))
    else:
        # Java JXRay layout: "Objects <live> <garbage> <total>" followed by
        # "Bytes <live>KB <garbage>KB <total>KB".
        match = re.search(r"Objects\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)", s5)
        bytes_match = re.search(
            r"Bytes\s+([\d,]+)KB\s+\([\d.]+%\)\s+([\d,]+)KB", s5
        )
        if match and bytes_match:
            values["live_objects"] = number(match.group(1))
            values["garbage_objects"] = number(match.group(2))
            values["live_bytes_kb"] = number(bytes_match.group(1))
            values["garbage_bytes_kb"] = number(bytes_match.group(2))

    for key, section in (
        ("s8", 8),
        ("s9", 9),
        ("s10", 10),
        ("s11", 11),
        ("s12", 12),
        ("s14", 14),
    ):
        text = section_text(doc, section)
        match = re.search(r"Overhead\s+([\d.]+)%\s*\(\s*([\d,]+)KB\s*\)", text)
        if not match:
            # Java's "not significant" sections omit the percentage.
            match = re.search(r"Overhead.*?([\d,]+)KB", text)
            if match:
                values[f"{key}_overhead_kb"] = number(match.group(1))
        else:
            values[f"{key}_overhead_kb"] = number(match.group(2))

    s18 = section_text(doc, 18)
    match = re.search(r"Off-heap\s+([\d,]+)KB", s18)
    if not match:
        # Java JXRay layout: "Size 54.4% ( 96,927KB )".
        match = re.search(r"Size\s+[\d.]+%\s*\(\s*([\d,]+)KB", s18)
    if match:
        values["s18_offheap_kb"] = number(match.group(1))
    match = re.search(r"unreachable.*?([\d,]+)KB\s*\(", s18, flags=re.S)
    if match:
        values["s18_unreachable_kb"] = number(match.group(1))

    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--baseline", default=str(Path(__file__).with_name("parity_baseline.json")))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="ignore baseline tolerances and require exact equality")
    args = parser.parse_args()

    doc = Path(args.report).read_text(encoding="utf-8", errors="replace")
    actual = extract(doc)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    rows = []
    failed = 0
    for name, spec in baseline.items():
        expected = spec["value"]
        tolerance = 0.0 if args.strict else spec.get("tolerance_pct", 0.0)
        got = actual.get(name)
        if got is None:
            status = "MISSING"
            delta_pct = None
        else:
            delta_pct = 0.0 if expected == 0 else (got - expected) * 100.0 / expected
            status = "PASS" if abs(delta_pct) <= tolerance else "FAIL"
        if status != "PASS":
            failed += 1
        rows.append((name, expected, got, delta_pct, tolerance, status))

    if args.json:
        print(json.dumps({name: actual.get(name) for name in baseline}, indent=2))
    else:
        print(f"{'metric':24} {'expected':>12} {'actual':>12} {'delta':>9} {'tol':>6}  status")
        for name, expected, got, delta_pct, tolerance, status in rows:
            delta_text = "-" if delta_pct is None else f"{delta_pct:+.2f}%"
            got_text = "-" if got is None else f"{got:,}"
            print(f"{name:24} {expected:>12,} {got_text:>12} {delta_text:>9} {tolerance:>5.1f}%  {status}")
        print(f"\n{len(rows) - failed}/{len(rows)} metrics within tolerance")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
