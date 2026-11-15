#!/usr/bin/env python3
"""Order-insensitive structural fingerprint of an SS-Mat HTML report.

Two runs of the same binary produce different bytes because several tables are
built from hash-map iteration, so ties come out in a different order. This tool
compares *content* instead: every table row and every section text line is
normalised and compared as a multiset.

Usage:
    python3 scripts/report_fingerprint.py REPORT            # print fingerprint digest
    python3 scripts/report_fingerprint.py A.html B.html    # compare, exit 1 on difference
"""
from __future__ import annotations

import hashlib
import html as html_lib
import re
import sys
from collections import Counter
from pathlib import Path


def normalise(text: str) -> str:
    text = html_lib.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def rows(doc: str) -> Counter:
    counter: Counter = Counter()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", doc, flags=re.S | re.I):
        cells = [normalise(re.sub(r"<[^>]+>", " ", c)) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)]
        if any(cells):
            counter[" | ".join(cells)] += 1
    return counter


def sections(doc: str) -> dict[str, Counter]:
    out: dict[str, Counter] = {}
    starts = [(m.group(1), m.start()) for m in re.finditer(r'id="Section-(\d+)"', doc)]
    for i, (num, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(doc)
        body = doc[start:end]
        body = re.sub(r"<script.*?</script>", " ", body, flags=re.S | re.I)
        body = re.sub(r"<style.*?</style>", " ", body, flags=re.S | re.I)
        body = re.sub(r"<[^>]+>", " ", body)
        counter: Counter = Counter()
        for line in normalise(body).split(" "):
            if line:
                counter[line] += 1
        out[num] = counter
    return out


def fingerprint(path: Path) -> dict:
    doc = path.read_text(encoding="utf-8", errors="replace")
    return {"rows": rows(doc), "sections": sections(doc)}


def digest(counter: Counter) -> str:
    h = hashlib.sha256()
    for key in sorted(counter):
        h.update(key.encode())
        h.update(b"\x00")
        h.update(str(counter[key]).encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    a = fingerprint(Path(sys.argv[1]))
    if len(sys.argv) == 2:
        print("rows:", digest(a["rows"]))
        for num in sorted(a["sections"], key=int):
            print(f"section {num:>2}: {digest(a['sections'][num])}")
        return 0

    b = fingerprint(Path(sys.argv[2]))
    failed = 0
    if a["rows"] != b["rows"]:
        failed += 1
        only_a = a["rows"] - b["rows"]
        only_b = b["rows"] - a["rows"]
        print(f"ROW MISMATCH: {sum(only_a.values())} rows only in A, {sum(only_b.values())} only in B")
        for row, n in list(only_a.items())[:5]:
            print(f"  A only x{n}: {row[:150]}")
        for row, n in list(only_b.items())[:5]:
            print(f"  B only x{n}: {row[:150]}")
    else:
        print("rows: identical multiset")
    for num in sorted(set(a["sections"]) | set(b["sections"]), key=int):
        ca = a["sections"].get(num, Counter())
        cb = b["sections"].get(num, Counter())
        if ca != cb:
            failed += 1
            diff_a = ca - cb
            diff_b = cb - ca
            print(f"section {num}: MISMATCH ({sum(diff_a.values())} tokens only in A, {sum(diff_b.values())} only in B)")
            for tok, n in list(diff_a.items())[:4]:
                print(f"    A only x{n}: {tok[:120]}")
            for tok, n in list(diff_b.items())[:4]:
                print(f"    B only x{n}: {tok[:120]}")
    if failed == 0:
        print("FINGERPRINT OK")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
