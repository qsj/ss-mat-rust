#!/usr/bin/env python3
"""Fail-closed baseline/candidate bundle comparison; no automatic nondeterminism waivers.

Bundles contain report.html and optionally debug/. Debug runs must use --debug.
Use --control with an independent old/old repeat to diagnose (not excuse) drift.
Use --html-diagnostic to decode |*/c<idx>|/|*/s<idx>| string ids before localizing
HTML differences per section; it is diagnostic and never alters the strict result.
Output contains counts and hashes only, never dump contents or secret values.
Exit 0: exact equivalence; 1: differences; 2: invalid/missing evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

GATES = {"problem_stats.tsv", "boxed_summary.tsv", "s8_summary.tsv",
         "s8_groups.tsv", "s15_dup_obj_arrays.tsv"}
# These are forensic sequences, not unordered aggregate relations.
ORDERED = {"ref_chains.tsv", "roots.tsv", "objects.tsv"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare_file(a: Path, b: Path, multiset: bool) -> dict:
    left, right = a.read_bytes(), b.read_bytes()
    result = {"equal": left == right, "left_sha256": digest(left),
              "right_sha256": digest(right), "mode": "bytes"}
    if multiset:
        # Retain exact cells, spaces, line endings, header and duplicate counts.
        ca, cb = Counter(left.splitlines(keepends=True)), Counter(right.splitlines(keepends=True))
        result.update(equal=ca == cb, mode="exact-line-multiset",
                      left_only=sum((ca - cb).values()), right_only=sum((cb - ca).values()))
    return result


def report_sections(data: bytes) -> dict[str, bytes]:
    marks = list(re.finditer(rb'id="Section-(\d+)"', data))
    result = {"preamble": data[:marks[0].start()] if marks else data}
    for i, mark in enumerate(marks):
        # Index also detects repeated or reordered sections. Keep scripts/attributes.
        result[f"{i}:{mark[1].decode()}"] = data[mark.start():marks[i + 1].start() if i + 1 < len(marks) else len(data)]
    return result


def compare(a: Path, b: Path, debug: bool) -> dict:
    for root in (a, b):
        if not (root / "report.html").is_file():
            raise ValueError("required report.html missing")
        if debug:
            for name in GATES:
                if not (root / "debug" / name).is_file():
                    raise ValueError(f"required debug file missing: {name}")
    files = {"report.html": compare_file(a / "report.html", b / "report.html", False)}
    sa = report_sections((a / "report.html").read_bytes())
    sb = report_sections((b / "report.html").read_bytes())
    files["report.html"]["changed_sections"] = [
        k for k in sorted(sa.keys() | sb.keys()) if sa.get(k) != sb.get(k)]
    if debug:
        da, db = a / "debug", b / "debug"
        names_a = {p.relative_to(da).as_posix() for p in da.rglob("*") if p.is_file()}
        names_b = {p.relative_to(db).as_posix() for p in db.rglob("*") if p.is_file()}
        for name in sorted(names_a | names_b):
            key = "debug/" + name
            if name not in names_a or name not in names_b:
                files[key] = {"equal": False, "missing": "baseline" if name not in names_a else "candidate"}
            else:
                files[key] = compare_file(da / name, db / name,
                                         name.endswith(".tsv") and Path(name).name not in ORDERED)
    return {"equal": all(v["equal"] for v in files.values()), "files": files}


def chain_diagnostic(a: Path, b: Path) -> dict:
    """Logical records can contain newlines; payload remains opaque exact bytes."""
    start = re.compile(rb"(?m)^(?:class\tclass=\d+\t|humongous\t)kind=(?:expensive|full)\tsize=-?\d+\tcount=\d+\tids=")

    def parse(path):
        data = path.read_bytes()
        marks = list(start.finditer(data))
        header = b"section\tclass\tkind\tsize\tcount\tids\tpath\tsamples\n"
        if not marks:
            if data == header:
                return []
            raise ValueError("unrecognized chain framing")
        if data[:marks[0].start()] != header:
            raise ValueError("unrecognized chain header")
        records = []
        for i, mark in enumerate(marks):
            raw = data[mark.start():marks[i + 1].start() if i + 1 < len(marks) else len(data)]
            if not raw.endswith(b"\n"):
                raise ValueError("unterminated chain record")
            structure, samples = raw[:-1].split(b"\tsamples=", 1)
            prefix, path_text = structure.split(b"\tpath=", 1)
            identity, ids = prefix.split(b"\tids=", 1)
            bucket, numbers = identity.split(b"\tsize=", 1)
            size, count = numbers.split(b"\tcount=", 1)
            records.append((bucket, path_text, int(size), int(count), ids, samples))
        return records

    left, right = parse(a), parse(b)
    la = {(x[0], x[1]): x for x in left}
    lb = {(x[0], x[1]): x for x in right}
    if len(la) != len(left) or len(lb) != len(right):
        raise ValueError("ambiguous duplicate chain keys")
    common = la.keys() & lb.keys()
    unmatched = []
    aggregates = []
    for bucket in sorted({x[0] for x in left + right}):
        aa = [x for x in left if x[0] == bucket]
        bb = [x for x in right if x[0] == bucket]
        def aggregate(rows):
            return [len(rows), sum(x[2] for x in rows), sum(x[3] for x in rows)]
        aggregates.append({"bucket_sha256": digest(bucket), "left": aggregate(aa), "right": aggregate(bb)})
        ua = [x for x in aa if (x[0], x[1]) not in lb]
        ub = [x for x in bb if (x[0], x[1]) not in la]
        if ua or ub:
            cut_a = min((x[2] for x in aa), default=None)
            cut_b = min((x[2] for x in bb), default=None)
            unmatched.append({"bucket_sha256": digest(bucket),
                              "left_key_sha256": [digest(x[0] + b"\t" + x[1]) for x in ua],
                              "right_key_sha256": [digest(x[0] + b"\t" + x[1]) for x in ub],
                              "left_sizes": [x[2] for x in ua], "right_sizes": [x[2] for x in ub],
                              "both_at_ten_group_cutoff": len(aa) == len(bb) == 10 and all(x[2] == cut_a == cut_b for x in ua + ub),
                              "status": "unresolved_selected_groups"})
    return {"logical_records": [len(left), len(right)], "common_keys": len(common),
            "common_keys_full_exact": all(la[k] == lb[k] for k in common),
            "aggregates_group_count_size_count": aggregates, "unmatched": unmatched,
            "framing_caveat": "Unescaped payloads can mimic record markers; ambiguous evidence requires manual review."}


def diagnostics(a: Path, b: Path) -> dict:
    result = {"policy": "Diagnostic only; never changes strict equality or exit status. Baseline variability is proven only by old/old controls."}
    try:
        result["ref_chains"] = chain_diagnostic(a / "debug/ref_chains.tsv", b / "debug/ref_chains.tsv")
        sa = (a / "debug/s14_summary.txt").read_bytes().splitlines(keepends=True)
        sb = (b / "debug/s14_summary.txt").read_bytes().splitlines(keepends=True)
        generated = lambda line: re.fullmatch(rb"generated: (?:unix=\d+s|unknown)\n", line) is not None
        result["s14_generated"] = {
            "generated_fields": [sum(map(generated, sa)), sum(map(generated, sb))],
            "other_bytes_equal": [x for x in sa if not generated(x)] == [x for x in sb if not generated(x)],
            "source": "baseline debug_dump.rs:290,355-358 SystemTime::now",
            "status": "unresolved_timestamp_difference" if sa != sb else "exact"}
        completeness = []
        import csv
        for root in (a, b):
            with (root / "debug/s8_groups.tsv").open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            missing = sum(not row["sample_obj_ids_hex"] and row["rep_obj_id_hex"] == "0x0" for row in rows)
            completeness.append({"groups": len(rows), "missing_representative_and_samples": missing,
                                 "status": "unresolved_missing_S8_detail" if missing else "no_placeholder_signature"})
        result["s8_completeness"] = completeness
    except (OSError, ValueError, KeyError, TypeError):
        result["diagnostic_error"] = "Unrecognized or incomplete evidence; strict comparison remains authoritative."
    return result


# ---- decoded-identifier HTML diagnosis (diagnostic only) ----
#
# The report replaces repeated class names and long scalar values inside
# reference-chain <details> blocks with |*/c<idx>| / |*/s<idx>| placeholders and
# emits the strings once, per run, in the ENCODED_STRINGS table. That table is
# built in first-registration order, so a semantically identical report can
# carry different indices. Decoding each report against its own table before
# comparing keeps identifier assignment separate from content. Nothing here
# changes compare(), its equality or the exit status.

ENCODED_START = "var ENCODED_STRINGS=["
ENCODED_END = "\nvar ESP="
ENCODED_ID = re.compile(r"\|*/([cs])(\d+)\|")
TEXT_SECTION_MARK = re.compile(r'id="Section-(\d+)"')
DETAILS_TAG = re.compile(r"</?details\b[^>]*>", re.I)
ROW = re.compile(r"<tr[^>]*>.*?</tr>", re.S)


def decode_encoded_ids(data: bytes) -> tuple[str, dict]:
    """Resolve |*/c<idx>| / |*/s<idx>| against the report's own table."""
    text = data.decode("utf-8", "surrogateescape")
    start = text.find(ENCODED_START)
    end = text.find(ENCODED_END, start) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError("encoded-string table not found")
    body = text[start + len(ENCODED_START):end].strip()
    if body.endswith("];"):
        body = body[:-2]
    body = body.strip()
    strings = json.loads("[" + body + "]") if body else []
    unresolved = [0]

    def replace(match):
        index = int(match.group(2))
        if index >= len(strings):
            unresolved[0] += 1
            return match.group(0)
        tag = "t-c" if match.group(1) == "c" else "t-s"
        return "<%s>%s</%s>" % (tag, strings[index], tag)

    decoded = ENCODED_ID.sub(replace, text)
    return decoded, {"count": len(strings), "unresolved": unresolved[0],
                     "digests": [digest(s.encode("utf-8", "surrogateescape")) for s in strings]}


def text_sections(text: str) -> dict[str, str]:
    """Same slicing policy as report_sections, applied to decoded text."""
    marks = list(TEXT_SECTION_MARK.finditer(text))
    sections = {"preamble": text[:marks[0].start()] if marks else text}
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections["%d:%s" % (i, mark.group(1))] = text[mark.start():end]
    return sections


def split_details(section: str) -> tuple[str, list[str]]:
    """Split a section into its skeleton and its top-level <details> payloads.

    Slices start at the id="Section-N" attribute inside the section's own
    <details> wrapper, so that wrapper's open tag is absent while its close tag
    is present (except for the last section).
    """
    skeleton = []
    blocks = []
    pos = 0
    depth = 1
    start = None
    for match in DETAILS_TAG.finditer(section):
        if match.group(0).startswith("</"):
            depth -= 1
            if depth == 0:
                skeleton.append(section[pos:match.start()])
                pos = match.end()
                break
            if depth == 1:
                blocks.append(section[start:match.end()])
                start = None
                pos = match.end()
        else:
            if depth == 1:
                skeleton.append(section[pos:match.start()])
                skeleton.append("<DETAILS>")
                pos = match.end()
                start = match.start()
            depth += 1
    skeleton.append(section[pos:])
    return "".join(skeleton), blocks


def summary_text(block: str) -> str:
    """The block's <summary> text, used for content-free identity hashing."""
    match = re.search(r"<summary>(.*?)</summary>", block, re.S)
    return re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else ""


def block_category(block: str) -> str:
    """Coarse label for one <details> payload; never returns its content."""
    match = re.search(r"<summary>(.*?)</summary>", block, re.S)
    title = re.sub(r"<[^>]+>", "", match.group(1)).lower() if match else ""
    for needle, label in (("typical reference paths", "typical-reference-paths"),
                          ("full reference chains", "full-reference-chains"),
                          ("per-instance reference chains", "per-instance-reference-chains"),
                          ("reference chains", "class-reference-chains"),
                          ("expensive data fields", "expensive-data-fields"),
                          ("object tree for gc root", "object-tree-gc-root"),
                          ("local variables", "local-variables"),
                          ("per-type breakdown", "per-type-breakdown")):
        if needle in title:
            return label
    return "other"


def section_diagnostic(x: str, y: str) -> dict:
    lines_a, lines_b = x.splitlines(keepends=True), y.splitlines(keepends=True)
    counts_a, counts_b = Counter(lines_a), Counter(lines_b)
    skeleton_a, blocks_a = split_details(x)
    skeleton_b, blocks_b = split_details(y)
    result = {
        "lines": [len(lines_a), len(lines_b)],
        "lines_multiset_equal": counts_a == counts_b,
        "lines_only_baseline": sum((counts_a - counts_b).values()),
        "lines_only_candidate": sum((counts_b - counts_a).values()),
        "skeleton_equal": skeleton_a == skeleton_b,
        "skeleton_bytes": [len(skeleton_a), len(skeleton_b)],
        "detail_blocks": [len(blocks_a), len(blocks_b)],
        "detail_blocks_differing": 0,
        "detail_blocks_order_only": 0,
        "detail_blocks_content": 0,
        "detail_blocks_non_row": 0,
        "detail_categories": {},
    }
    if len(blocks_a) != len(blocks_b):
        result["detail_blocks_comparable"] = False
        return result
    categories = Counter()
    for block_a, block_b in zip(blocks_a, blocks_b):
        if block_a == block_b:
            continue
        result["detail_blocks_differing"] += 1
        categories[block_category(block_a)] += 1
        rows_a, rows_b = ROW.findall(block_a), ROW.findall(block_b)
        if Counter(rows_a) != Counter(rows_b):
            result["detail_blocks_content"] += 1
        elif rows_a != rows_b:
            result["detail_blocks_order_only"] += 1
        else:
            result["detail_blocks_non_row"] += 1
    result["detail_categories"] = dict(sorted(categories.items()))
    # Payload identity without content: hash each block summary. An equal
    # multiset means the same payloads were rendered, possibly reordered; an
    # unequal one means the selection itself differs (e.g. a tie cutoff).
    summaries_a = Counter(digest(summary_text(b).encode("utf-8", "surrogateescape")) for b in blocks_a)
    summaries_b = Counter(digest(summary_text(b).encode("utf-8", "surrogateescape")) for b in blocks_b)
    result["detail_summaries_multiset_equal"] = summaries_a == summaries_b
    result["detail_summaries_only_baseline"] = sum((summaries_a - summaries_b).values())
    result["detail_summaries_only_candidate"] = sum((summaries_b - summaries_a).values())
    skeleton_lines_a = Counter(skeleton_a.splitlines(keepends=True))
    skeleton_lines_b = Counter(skeleton_b.splitlines(keepends=True))
    result["skeleton_lines_only_baseline"] = sum((skeleton_lines_a - skeleton_lines_b).values())
    result["skeleton_lines_only_candidate"] = sum((skeleton_lines_b - skeleton_lines_a).values())
    return result


def preamble_diagnostic(x: str, y: str) -> dict:
    """Localize preamble drift: head, encoded-string table, decoded tail."""
    def split(text):
        start = text.find(ENCODED_START)
        end = text.find(ENCODED_END, start) if start >= 0 else -1
        if start < 0 or end < 0:
            return None
        return text[:start + len(ENCODED_START)], text[start + len(ENCODED_START):end], text[end:]
    left, right = split(x), split(y)
    if left is None or right is None:
        return {"preamble_split": "unrecognized"}
    return {"head_before_encoded_array_equal": left[0] == right[0],
            "encoded_array_text_equal": left[1] == right[1],
            "tail_after_encoded_array_equal": left[2] == right[2]}


def html_diagnostic(a: Path, b: Path) -> dict:
    raw_a, raw_b = (a / "report.html").read_bytes(), (b / "report.html").read_bytes()
    text_a, meta_a = decode_encoded_ids(raw_a)
    text_b, meta_b = decode_encoded_ids(raw_b)
    sections_a, sections_b = text_sections(text_a), text_sections(text_b)
    raw_a_sections, raw_b_sections = report_sections(raw_a), report_sections(raw_b)
    sections = {}
    for key in sorted(sections_a.keys() | sections_b.keys()):
        x, y = sections_a.get(key, ""), sections_b.get(key, "")
        if x == y:
            sections[key] = {"decoded_equal": True}
            continue
        entry = {"decoded_equal": False}
        entry.update(preamble_diagnostic(x, y) if key == "preamble" else section_diagnostic(x, y))
        sections[key] = entry
    return {
        "policy": "Diagnostic only; decoding encoded ids separates identifier assignment from "
                  "content. Strict comparison and exit status are unchanged.",
        "encoded_strings": {
            "counts": [meta_a["count"], meta_b["count"]],
            "unresolved_placeholders": [meta_a["unresolved"], meta_b["unresolved"]],
            "sequence_equal": meta_a["digests"] == meta_b["digests"],
            "multiset_equal": Counter(meta_a["digests"]) == Counter(meta_b["digests"]),
        },
        "raw_report_equal": raw_a == raw_b,
        "decoded_report_equal": text_a == text_b,
        "sections_raw_differing": [k for k in sorted(raw_a_sections.keys() | raw_b_sections.keys())
                                   if raw_a_sections.get(k) != raw_b_sections.get(k)],
        "sections_decoded_differing": [k for k, v in sections.items() if not v["decoded_equal"]],
        "sections": sections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--control", type=Path, help="old/old repeat; diagnostic only")
    parser.add_argument("--html-diagnostic", action="store_true",
                        help="decode encoded string ids and localize HTML drift per section")
    args = parser.parse_args()
    try:
        result = {"comparison": compare(args.baseline, args.candidate, args.debug)}
        if args.debug:
            result["diagnostics"] = diagnostics(args.baseline, args.candidate)
        if args.html_diagnostic:
            result["html_diagnostic"] = html_diagnostic(args.baseline, args.candidate)
            if args.control:
                result["html_control_diagnostic"] = html_diagnostic(args.baseline, args.control)
        if args.control:
            result["old_old_control"] = compare(args.baseline, args.control, args.debug)
            if args.debug:
                result["old_old_diagnostics"] = diagnostics(args.baseline, args.control)
            result["control_policy"] = "Differences remain failures; manual attribution required, no automatic exclusions."
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["comparison"]["equal"] else 1
    except (OSError, ValueError) as error:
        # Avoid exposing input values or contents from exceptions.
        print(json.dumps({"error": str(error) if isinstance(error, ValueError) else type(error).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
