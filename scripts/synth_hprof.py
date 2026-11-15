#!/usr/bin/env python3
"""Write a synthetic, valid HPROF 1.0.2 heap dump holding a chosen number of objects.

The generator exists so memory growth can be measured at two heap sizes instead of
on the single reference dump. The byte layout is dictated by the two readers that
must both accept the file:

* crates/hprof-parser/src/record_reader.rs -- the strict raw record walker this
  codebase re-parses the file with (every top-level record must be consumed
  exactly, every heap-dump sub-record must be a known tag);
* jvm-hprof 0.1 -- the crate that performs the record-level parse, which panics
  on an unknown top-level tag and asserts the ClassDump constant-pool length is 0.

Layout: HPROF header, UTF8 records for every name, one LOAD_CLASS per class, then
HEAP_DUMP_SEGMENT (0x1C) records carrying CLASS_DUMP (0x20), INSTANCE_DUMP (0x21),
OBJECT_ARRAY_DUMP (0x22) and PRIMITIVE_ARRAY_DUMP (0x23) sub-records plus GC
roots, and a final HEAP_DUMP_END (0x2C). Every top-level record is
"u1 tag, u4 time_delta, u4 body_len, body".

Objects are chained through the reference field in runs of --chain, so the BFS has
a live object graph of realistic depth and nothing is left dangling.

Usage:
    python3 scripts/synth_hprof.py --objects 1000000 --out /tmp/synth-1m.hprof
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

ID_SIZE = 8
MAGIC = b"JAVA PROFILE 1.0.2\x00"

TAG_UTF8 = 0x01
TAG_LOAD_CLASS = 0x02
TAG_HEAP_DUMP_SEGMENT = 0x1C
TAG_HEAP_DUMP_END = 0x2C

SR_ROOT_JNI_GLOBAL = 0x01
SR_ROOT_STICKY_CLASS = 0x05
SR_ROOT_THREAD_BLOCK = 0x06
SR_ROOT_UNKNOWN = 0xFF
SR_CLASS_DUMP = 0x20
SR_INSTANCE_DUMP = 0x21
SR_OBJ_ARRAY_DUMP = 0x22
SR_PRIM_ARRAY_DUMP = 0x23

# HPROF basic type codes (record_reader::field_width / jvm_hprof::FieldType).
TY_OBJECT = 2
TY_BYTE = 8
TY_INT = 10

# class table: (name, string_id, class_obj_id, load_class_serial, super_obj_id,
#               instance_size, instance_fields, static_fields)
# demo.Node declares exactly the three fields an INSTANCE_DUMP body carries, so the
# field walker in the analyser sees a body whose length matches its descriptors.
CLASSES = [
    ("java.lang.Object", 1, 0x1001, 1, 0, 16, [], []),
    (
        "demo.Node",
        2,
        0x1002,
        2,
        0x1001,
        24,
        [(6, TY_OBJECT), (7, TY_BYTE), (8, TY_INT)],
        [(9, TY_INT, struct.pack(">i", 42))],
    ),
    ("byte[]", 3, 0x1003, 3, 0, 0, [], []),
    ("int[]", 4, 0x1004, 4, 0, 0, [], []),
    ("demo.Node[]", 5, 0x1005, 5, 0, 0, [], []),
]
STRINGS = {
    1: "java.lang.Object",
    2: "demo.Node",
    3: "byte[]",
    4: "int[]",
    5: "demo.Node[]",
    6: "next",
    7: "flag",
    8: "value",
    9: "COUNT",
}
NODE_CLASS_OBJ_ID = 0x1002
NODE_ARRAY_CLASS_OBJ_ID = 0x1005

# INSTANCE_DUMP body for demo.Node: next (id) + flag (byte) + value (int).
INSTANCE_BODY_LEN = ID_SIZE + 1 + 4
# tag + addr + stack_serial + class id + fields_byte_len + body
INSTANCE_RECORD_LEN = 1 + ID_SIZE + 4 + ID_SIZE + 4 + INSTANCE_BODY_LEN
INSTANCE_RECORD = ">BQIQIQBi"

INSTANCE_BASE = 0x0000_0001_0000_0000
INSTANCE_STRIDE = 0x20
PRIM_ARRAY_BASE = 0x0000_0002_0000_0000
OBJ_ARRAY_ID = 0x0000_0003_0000_0000

PRIM_ARRAYS = [
    (PRIM_ARRAY_BASE + 1, TY_BYTE, b"\x00" * 16, 1),
    (PRIM_ARRAY_BASE + 2, TY_BYTE, b"ss-mat synthetic heap dump\x00", 1),
    (PRIM_ARRAY_BASE + 3, TY_BYTE, bytes(range(64)), 1),
    (PRIM_ARRAY_BASE + 4, TY_BYTE, b"\xff" * 1024, 1),
    (PRIM_ARRAY_BASE + 5, TY_INT, struct.pack(">8i", *range(8)), 4),
    (PRIM_ARRAY_BASE + 6, TY_INT, b"\x00" * 4, 4),
]
OBJ_ARRAY_ELEMS = 4


def instance_id(index: int) -> int:
    return INSTANCE_BASE + index * INSTANCE_STRIDE


def top_record(out, tag: int, body: bytes) -> None:
    out.write(struct.pack(">BII", tag, 0, len(body)))
    out.write(body)


def utf8_body(name_id: int, text: str) -> bytes:
    return struct.pack(">Q", name_id) + text.encode("utf-8")


def load_class_body(serial: int, class_obj_id: int, name_id: int) -> bytes:
    # u4 class_serial, id class_obj_id, u4 stack_trace_serial, id class_name_id
    return struct.pack(">IQIQ", serial, class_obj_id, 0, name_id)


def class_dump(class_obj_id, super_id, instance_size, inst_fields, statics) -> bytes:
    parts = [
        struct.pack(">B", SR_CLASS_DUMP),
        struct.pack(">Q", class_obj_id),
        struct.pack(">I", 0),  # stack trace serial
        struct.pack(">Q", super_id),
        struct.pack(">Q", 0),  # class loader
        struct.pack(">Q", 0),  # signers
        struct.pack(">Q", 0),  # protection domain
        struct.pack(">Q", 0),  # reserved
        struct.pack(">Q", 0),  # reserved
        struct.pack(">I", instance_size),
        # The constant pool is always empty in real dumps and jvm-hprof asserts it.
        struct.pack(">H", 0),
        struct.pack(">H", len(statics)),
    ]
    for name_id, ty, value in statics:
        parts.append(struct.pack(">QB", name_id, ty))
        parts.append(value)
    parts.append(struct.pack(">H", len(inst_fields)))
    for name_id, ty in inst_fields:
        parts.append(struct.pack(">QB", name_id, ty))
    return b"".join(parts)


def prim_array_record(addr: int, ty: int, payload: bytes, width: int) -> bytes:
    head = struct.pack(">BQIIB", SR_PRIM_ARRAY_DUMP, addr, 0, len(payload) // width, ty)
    return head + payload


def obj_array_record(addr: int, class_obj_id: int, elements) -> bytes:
    head = struct.pack(">BQIIQ", SR_OBJ_ARRAY_DUMP, addr, 0, len(elements), class_obj_id)
    return head + b"".join(struct.pack(">Q", e) for e in elements)


def root_sticky_class(class_obj_id: int) -> bytes:
    return struct.pack(">BQ", SR_ROOT_STICKY_CLASS, class_obj_id)


def root_jni_global(obj_id: int, global_ref_id: int) -> bytes:
    return struct.pack(">BQQ", SR_ROOT_JNI_GLOBAL, obj_id, global_ref_id)


def root_thread_block(obj_id: int, thread_serial: int) -> bytes:
    return struct.pack(">BQI", SR_ROOT_THREAD_BLOCK, obj_id, thread_serial)


def root_unknown(obj_id: int) -> bytes:
    return struct.pack(">BQ", SR_ROOT_UNKNOWN, obj_id)


def write_header(out, timestamp_millis: int) -> None:
    out.write(MAGIC)
    out.write(struct.pack(">I", ID_SIZE))
    out.write(struct.pack(">Q", timestamp_millis))


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a synthetic HPROF 1.0.2 dump.")
    parser.add_argument("--objects", type=int, required=True,
                        help="number of INSTANCE_DUMP objects")
    parser.add_argument("--out", required=True, help="output .hprof path")
    parser.add_argument("--segment-objects", type=int, default=100000,
                        help="instances per HEAP_DUMP_SEGMENT record (default 100000)")
    parser.add_argument("--chain", type=int, default=1000,
                        help="reference-chain length that keeps objects live (default 1000)")
    parser.add_argument("--timestamp", type=int, default=1700000000000,
                        help="HPROF header timestamp in millis")
    args = parser.parse_args()

    if args.objects < 1:
        parser.error("--objects must be >= 1")
    if args.chain < 1:
        parser.error("--chain must be >= 1")
    if args.segment_objects < 1:
        parser.error("--segment-objects must be >= 1")

    n = args.objects
    chain = args.chain
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as out:
        write_header(out, args.timestamp)

        for name_id in sorted(STRINGS):
            top_record(out, TAG_UTF8, utf8_body(name_id, STRINGS[name_id]))

        for name, name_id, class_obj_id, serial, super_id, _sz, _f, _s in CLASSES:
            top_record(out, TAG_LOAD_CLASS, load_class_body(serial, class_obj_id, name_id))

        # First segment: every class dump, the class roots, the primitive arrays and
        # the object array. Classes must precede the instances that reference them.
        head = bytearray()
        for name, _name_id, class_obj_id, _serial, super_id, size, fields, statics in CLASSES:
            head += class_dump(class_obj_id, super_id, size, fields, statics)
        for _name, _name_id, class_obj_id, _serial, *_rest in CLASSES:
            head += root_sticky_class(class_obj_id)
        head += root_jni_global(instance_id(0), 0x7FFF000000000001)
        head += root_thread_block(instance_id(0), 1)
        for addr, ty, payload, width in PRIM_ARRAYS:
            head += prim_array_record(addr, ty, payload, width)
        head += obj_array_record(
            OBJ_ARRAY_ID,
            NODE_ARRAY_CLASS_OBJ_ID,
            [instance_id(i) for i in range(OBJ_ARRAY_ELEMS)],
        )
        top_record(out, TAG_HEAP_DUMP_SEGMENT, bytes(head))

        # Instance segments: one INSTANCE_DUMP per object, the reference field chaining
        # each run of chain objects so the whole heap is live from a small root set.
        written = 0
        while written < n:
            count = min(args.segment_objects, n - written)
            buf = bytearray(count * INSTANCE_RECORD_LEN)
            offset = 0
            for i in range(written, written + count):
                addr = instance_id(i)
                link = i % chain
                ref = instance_id(i + 1) if link != chain - 1 and i + 1 < n else 0
                struct.pack_into(
                    INSTANCE_RECORD,
                    buf,
                    offset,
                    SR_INSTANCE_DUMP,
                    addr,
                    0,
                    NODE_CLASS_OBJ_ID,
                    INSTANCE_BODY_LEN,
                    ref,
                    i & 0xFF,
                    i & 0x7FFFFFFF,
                )
                offset += INSTANCE_RECORD_LEN
            top_record(out, TAG_HEAP_DUMP_SEGMENT, bytes(buf))
            written += count

        # Chain heads are the only strong roots into the instance graph.
        roots = bytearray()
        for i in range(0, n, chain):
            roots += root_unknown(instance_id(i))
        top_record(out, TAG_HEAP_DUMP_SEGMENT, bytes(roots))

        top_record(out, TAG_HEAP_DUMP_END, b"")

    objects_total = n + len(PRIM_ARRAYS) + 1
    size = out_path.stat().st_size
    print(
        "synth_hprof: instances=%d prim_arrays=%d obj_arrays=1 objects_total=%d bytes=%d out=%s"
        % (n, len(PRIM_ARRAYS), objects_total, size, out_path)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
