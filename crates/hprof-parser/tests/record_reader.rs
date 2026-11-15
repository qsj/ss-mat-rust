//! Tests for the raw HPROF record walker, built from synthetic byte arrays plus one
//! opt-in walk over a real dump (`SS_MAT_BIG_DUMP`).

use ss_hprof_parser::record_reader::{
    walk_objects, ObjRec, KIND_CLASS, KIND_INSTANCE, KIND_OBJ_ARRAY, KIND_PRIM_ARRAY,
};

const HEAP_DUMP: u8 = 0x0C;
const HEAP_DUMP_SEGMENT: u8 = 0x1C;
const MAGIC: &[u8] = b"JAVA PROFILE 1.0.";

/// Big-endian byte emitter; `pos()` is the offset of the next byte to be written.
struct Emit {
    buf: Vec<u8>,
    id_size: u8,
}

impl Emit {
    fn new(id_size: u8) -> Self {
        Emit { buf: Vec::new(), id_size }
    }

    /// A complete HPROF 1.0.2 file header for `id_size`.
    fn file(id_size: u8) -> Self {
        let mut e = Emit::new(id_size);
        e.raw(b"JAVA PROFILE 1.0.2\0");
        e.u32(id_size as u32);
        e.u64(1_700_000_000_000);
        e
    }

    fn pos(&self) -> u64 {
        self.buf.len() as u64
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn id(&mut self, v: u64) {
        if self.id_size == 4 {
            self.u32(v as u32);
        } else {
            self.u64(v);
        }
    }

    fn zeros(&mut self, n: u64) {
        self.buf.resize(self.buf.len() + n as usize, 0);
    }
}

/// Append a top-level record and return the absolute offset of its body.
fn record(file: &mut Emit, tag: u8, body: &Emit) -> u64 {
    file.u8(tag);
    file.u32(0); // time delta
    file.u32(body.buf.len() as u32);
    let body_at = file.pos();
    file.raw(&body.buf);
    body_at
}

fn header_len(data: &[u8]) -> usize {
    assert!(data.starts_with(MAGIC), "fixture must start with the HPROF magic");
    let version_and_nul = 2;
    let id_size_at = MAGIC.len() + version_and_nul;
    id_size_at + 4 + 8
}

#[derive(Debug)]
struct Fixture {
    data: Vec<u8>,
    id_size: u8,
    /// Offsets relative to the heap-dump body that the sub-records live in.
    super_aux: u64,
    child_aux: u64,
    instance_body: u64,
    instance_bytes: u64,
    obj_array_body: u64,
    prim_array_body: u64,
}

/// One heap-dump record holding: a superclass class dump, a subclass class dump with a
/// constant-pool entry, a static field and two instance fields, then one instance, one
/// object array and one primitive array. A second record holds the GC roots, and an
/// unknown top-level tag sits in front of both to prove it is skipped by length.
fn build(id_size: u8) -> Fixture {
    let mut sub = Emit::new(id_size);

    // CLASS_DUMP for the superclass (obj id 0x1000) declaring one int instance field.
    sub.u8(0x20);
    let super_aux = sub.pos();
    sub.id(0x1000);
    sub.u32(7);
    sub.id(0); // super
    sub.id(0); // class loader
    sub.id(0); // signers
    sub.id(0); // protection domain
    sub.id(0); // reserved
    sub.id(0); // reserved
    sub.u32(16);
    sub.u16(0); // constant pool
    sub.u16(0); // static fields
    sub.u16(1); // instance fields
    sub.id(0x11);
    sub.u8(10); // int

    // CLASS_DUMP for the subclass (obj id 0x2000).
    sub.u8(0x20);
    let child_aux = sub.pos();
    sub.id(0x2000);
    sub.u32(8);
    sub.id(0x1000); // super
    sub.id(0x3000);
    sub.id(0x3001);
    sub.id(0x3002);
    sub.id(0);
    sub.id(0);
    sub.u32(32);
    sub.u16(1); // one constant pool entry
    sub.u16(1);
    sub.u8(2); // object type
    sub.id(0x5000);
    sub.u16(1); // one static field
    sub.id(0x21);
    sub.u8(11); // long
    sub.u64(0x1122_3344_5566_7788);
    sub.u16(2); // two instance fields
    sub.id(0x22);
    sub.u8(4); // boolean
    sub.id(0x23);
    sub.u8(2); // object reference
    // declared bytes: 1 (boolean) + id_size (ref); the superclass adds 4 (int).

    // INSTANCE_DUMP for obj 0x9000 of class 0x2000.
    let instance_bytes = 1 + id_size as u64 + 4;
    sub.u8(0x21);
    sub.id(0x9000);
    sub.u32(9);
    sub.id(0x2000);
    sub.u32(instance_bytes as u32);
    let instance_body = sub.pos();
    sub.zeros(instance_bytes);

    // OBJECT_ARRAY_DUMP for obj 0xA000 with three elements of array class 0xB000.
    sub.u8(0x22);
    sub.id(0xA000);
    sub.u32(10);
    sub.u32(3);
    sub.id(0xB000);
    let obj_array_body = sub.pos();
    sub.zeros(3 * id_size as u64);

    // PRIMITIVE_ARRAY_DUMP for obj 0xC000, four chars.
    sub.u8(0x23);
    sub.id(0xC000);
    sub.u32(11);
    sub.u32(4);
    sub.u8(5); // char
    let prim_array_body = sub.pos();
    sub.zeros(8);

    let mut file = Emit::file(id_size);

    // An unknown top-level tag whose body must be skipped by length.
    let mut junk = Emit::new(id_size);
    junk.raw(b"skip exactly this");
    record(&mut file, 0x55, &junk);

    let base = record(&mut file, HEAP_DUMP, &sub);

    // Roots live in a second heap-dump record, this time a segment.
    let mut roots = Emit::new(id_size);
    roots.u8(0xFF);
    roots.id(0x7000);
    roots.u8(0x01);
    roots.id(0x7001);
    roots.id(0x7002);
    roots.u8(0x02);
    roots.id(0x7003);
    roots.u32(1);
    roots.u32(2);
    roots.u8(0x08);
    roots.id(0x7004);
    roots.u32(3);
    roots.u32(4);
    record(&mut file, HEAP_DUMP_SEGMENT, &roots);

    Fixture {
        data: file.buf,
        id_size,
        super_aux: base + super_aux,
        child_aux: base + child_aux,
        instance_body: base + instance_body,
        instance_bytes,
        obj_array_body: base + obj_array_body,
        prim_array_body: base + prim_array_body,
    }
}

fn walk(fixture: &Fixture) -> (Vec<ObjRec>, ss_hprof_parser::record_reader::WalkStats) {
    let mut recs = Vec::new();
    let stats = walk_objects(&fixture.data, fixture.id_size, |r| recs.push(*r))
        .unwrap_or_else(|e| panic!("walk failed: {e:#}"));
    (recs, stats)
}

fn case(id_size: u8) {
    let fixture = build(id_size);
    // Field bytes of the single instance in the fixture: one reference, one byte, one int.
    let instance_fields_len: u64 = id_size as u64 + 1 + 4;
    let (recs, stats) = walk(&fixture);

    assert_eq!(stats.instances, 1, "id_size {id_size}");
    assert_eq!(stats.object_arrays, 1, "id_size {id_size}");
    assert_eq!(stats.prim_arrays, 1, "id_size {id_size}");
    assert_eq!(stats.class_dumps, 2, "id_size {id_size}");
    assert_eq!(stats.gc_roots, 4, "id_size {id_size}");
    assert_eq!(recs.len(), 5, "id_size {id_size}: {recs:#?}");

    let mut super_rec = recs[0];
    assert!(super_rec.body_len > 0, "class body length is reported");
    super_rec.body_len = 0;
    assert_eq!(
        super_rec,
        ObjRec {
            addr: 0x1000,
            kind: KIND_CLASS,
            aux_offset: fixture.super_aux,
            ..ObjRec::default()
        },
        "superclass dump"
    );
    let mut child_rec = recs[1];
    assert!(child_rec.body_len > 0, "class body length is reported");
    child_rec.body_len = 0;
    assert_eq!(
        child_rec,
        ObjRec {
            addr: 0x2000,
            kind: KIND_CLASS,
            aux_offset: fixture.child_aux,
            ..ObjRec::default()
        },
        "subclass dump"
    );
    assert_eq!(
        recs[2],
        ObjRec {
            addr: 0x9000,
            kind: KIND_INSTANCE,
            class_ref: 0x2000,
            body_offset: fixture.instance_body,
            body_len: instance_fields_len as u32,
            ..ObjRec::default()
        },
        "instance dump"
    );
    assert_eq!(
        recs[3],
        ObjRec {
            addr: 0xA000,
            kind: KIND_OBJ_ARRAY,
            class_ref: 0xB000,
            elem_size: id_size,
            len: 3,
            body_offset: fixture.obj_array_body,
            body_len: 3 * id_size as u32,
            ..ObjRec::default()
        },
        "object array dump"
    );
    assert_eq!(
        recs[4],
        ObjRec {
            addr: 0xC000,
            kind: KIND_PRIM_ARRAY,
            prim_type: 5,
            elem_size: 2,
            len: 4,
            body_offset: fixture.prim_array_body,
            body_len: 8,
            ..ObjRec::default()
        },
        "primitive array dump"
    );

    // The instance body must be followed immediately by the object array sub-record: the
    // body length is what keeps the walk aligned.
    let next = (fixture.instance_body + fixture.instance_bytes) as usize;
    assert_eq!(fixture.data[next], 0x22, "instance body length must be exact");

    // Offsets are true file offsets, so they must all point inside the file.
    for rec in &recs {
        assert!(rec.body_offset < fixture.data.len() as u64);
        if rec.kind == KIND_CLASS {
            assert!(rec.aux_offset < fixture.data.len() as u64);
        }
    }

    // The same record area without the header yields the same objects, shifted by the header.
    let start = header_len(&fixture.data);
    let mut shifted = Vec::new();
    let body_only_stats =
        walk_objects(&fixture.data[start..], id_size, |r| shifted.push(*r)).expect("body-only walk");
    assert_eq!(body_only_stats, stats);
    assert_eq!(shifted.len(), recs.len());
    for (a, b) in shifted.iter().zip(recs.iter()) {
        assert_eq!(a.addr, b.addr);
        assert_eq!(a.kind, b.kind);
        let shift = start as u64;
        let expected_body = if b.kind == KIND_CLASS { 0 } else { b.body_offset - shift };
        let expected_aux = if b.kind == KIND_CLASS { b.aux_offset - shift } else { 0 };
        assert_eq!(a.body_offset, expected_body);
        assert_eq!(a.aux_offset, expected_aux);
    }
}

#[test]
fn walks_synthetic_dump_with_id_size_4() {
    case(4);
}

#[test]
fn walks_synthetic_dump_with_id_size_8() {
    case(8);
}

fn error_fixture(id_size: u8, sub: &[u8]) -> Vec<u8> {
    let mut body = Emit::new(id_size);
    body.raw(sub);
    let mut file = Emit::file(id_size);
    record(&mut file, HEAP_DUMP, &body);
    file.buf
}

#[test]
fn rejects_unknown_sub_record_tag() {
    let data = error_fixture(8, &[0x42, 0, 0, 0, 0]);
    let err = walk_objects(&data, 8, |_| {}).unwrap_err().to_string();
    assert!(err.contains("0x42"), "unexpected error: {err}");
    assert!(err.contains("offset"), "unexpected error: {err}");
}

#[test]
fn rejects_unknown_field_type() {
    let mut sub = Emit::new(8);
    sub.u8(0x20);
    sub.id(0x1000);
    sub.u32(0);
    sub.id(0);
    sub.id(0);
    sub.id(0);
    sub.id(0);
    sub.id(0);
    sub.id(0);
    sub.u32(16);
    sub.u16(0);
    sub.u16(1); // one static field of an impossible type
    sub.id(0x11);
    sub.u8(12);
    let data = error_fixture(8, &sub.buf);
    let err = walk_objects(&data, 8, |_| {}).unwrap_err().to_string();
    assert!(err.contains("unknown HPROF field type 12"), "unexpected error: {err}");
}

#[test]
fn rejects_truncated_record_body() {
    let mut file = Emit::file(4);
    file.u8(HEAP_DUMP);
    file.u32(0);
    file.u32(64); // body_len far beyond the end of the data
    file.raw(&[0x20, 0, 0, 0, 0]);
    let err = walk_objects(&file.buf, 4, |_| {}).unwrap_err().to_string();
    assert!(err.contains("runs past the end of data"), "unexpected error: {err}");
}

#[test]
fn rejects_heap_body_drift() {
    // An object array that claims three 8-byte elements inside a body with room for one.
    // A trailer record keeps the overshoot inside the file, so the walker reports drift
    // instead of a truncation.
    let mut sub = Emit::new(8);
    sub.u8(0x22);
    sub.id(0xA000);
    sub.u32(0);
    sub.u32(3);
    sub.id(0xB000);
    sub.zeros(8);

    let mut body = Emit::new(8);
    body.raw(&sub.buf);
    let mut file = Emit::file(8);
    record(&mut file, HEAP_DUMP, &body);
    let mut trailer = Emit::new(8);
    trailer.zeros(64);
    record(&mut file, 0x55, &trailer);

    let err = walk_objects(&file.buf, 8, |_| {}).unwrap_err().to_string();
    assert!(err.contains("drift"), "unexpected error: {err}");
}

#[test]
fn rejects_id_size_mismatch_with_header() {
    let fixture = build(8);
    let err = walk_objects(&fixture.data, 4, |_| {}).unwrap_err().to_string();
    assert!(err.contains("does not match the HPROF header"), "unexpected error: {err}");
}

#[test]
fn rejects_impossible_id_size() {
    let err = walk_objects(&[], 5, |_| {}).unwrap_err().to_string();
    assert!(err.contains("id_size must be 4 or 8"), "unexpected error: {err}");
}

// ---------------------------------------------------------------------------
// Real dump (opt-in)
// ---------------------------------------------------------------------------

/// End of the top-level record chain, computed independently of the walker.
fn record_chain_end(data: &[u8]) -> usize {
    let mut pos = header_len(data);
    while pos < data.len() {
        let mut len = [0u8; 4];
        len.copy_from_slice(&data[pos + 5..pos + 9]);
        pos += 9 + u32::from_be_bytes(len) as usize;
    }
    pos
}

#[test]
fn walks_real_dump_when_requested() {
    let Ok(path) = std::env::var("SS_MAT_BIG_DUMP") else {
        eprintln!("SS_MAT_BIG_DUMP not set, skipping the real-dump walk");
        return;
    };
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));

    let id_size_at = header_len(&data) - 12;
    let id_size = u32::from_be_bytes([
        data[id_size_at],
        data[id_size_at + 1],
        data[id_size_at + 2],
        data[id_size_at + 3],
    ]) as u8;

    let started = std::time::Instant::now();
    let mut objects = 0u64;
    let stats = walk_objects(&data, id_size, |_| objects += 1)
        .unwrap_or_else(|e| panic!("walk of {path} failed: {e:#}"));
    let elapsed = started.elapsed();

    let megabytes = data.len() as f64 / (1024.0 * 1024.0);
    println!(
        "walked {path}: id_size={id_size}, {} objects ({objects} recs), {} bytes ({megabytes:.1} MiB) in {elapsed:?} ({:.1} MiB/s)",
        stats.instances + stats.object_arrays + stats.prim_arrays + stats.class_dumps,
        data.len(),
        megabytes / elapsed.as_secs_f64()
    );
    println!("{stats:?}");

    assert_eq!(stats.instances, 2_135_318);
    assert_eq!(stats.object_arrays, 247_623);
    assert_eq!(stats.prim_arrays, 382_570);
    // 29,169 CLASS_DUMP sub-records. The jvm-hprof pipeline reports 29,177 classes because
    // `resolve_class_by_name` synthesizes one class per primitive array type ("byte[]", ...)
    // and HPROF names those classes "[B", so all 8 are always extra.
    assert_eq!(stats.class_dumps, 29_169);
    assert!(stats.gc_roots > 0, "expected GC roots, got {stats:?}");
    assert_eq!(
        objects,
        stats.instances + stats.object_arrays + stats.prim_arrays + stats.class_dumps,
        "one ObjRec per object"
    );
    assert_eq!(
        record_chain_end(&data),
        data.len(),
        "the walk must end exactly at EOF"
    );
}
