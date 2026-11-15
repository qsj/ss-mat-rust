//! Raw HPROF record walker.
//!
//! jvm-hprof parses records but exposes neither record byte offsets nor a way to
//! seek into an object body, so the memory-bounded pipeline needs its own walker
//! that reports, for every heap-dump object, where its body starts in the file.
//!
//! The walker is strict: every top-level record must be consumed exactly. A
//! mismatch means the sub-record layout below is wrong, and the error names the
//! offset so it can be fixed instead of silently drifting.
//!
//! Sub-record layouts mirror HotSpot's heapDumper.cpp as decoded by jvm-hprof 0.1:
//! INSTANCE_DUMP carries an explicit u4 field-bytes length, so no class-hierarchy
//! resolution is needed to know where the next sub-record starts.

use anyhow::{anyhow, bail, Result};

/// Object kinds reported by walk_objects.
pub const KIND_INSTANCE: u8 = 0;
pub const KIND_OBJ_ARRAY: u8 = 1;
pub const KIND_PRIM_ARRAY: u8 = 2;
pub const KIND_CLASS: u8 = 3;

/// Tag of a HEAP_DUMP record.
pub const TAG_HEAP_DUMP: u8 = 0x0C;
/// Tag of a HEAP_DUMP_SEGMENT record.
pub const TAG_HEAP_DUMP_SEGMENT: u8 = 0x1C;

/// One heap-dump object located in the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ObjRec {
    /// Object id (the HPROF address used as the heap-wide key).
    pub addr: u64,
    /// One of KIND_INSTANCE, KIND_OBJ_ARRAY, KIND_PRIM_ARRAY, KIND_CLASS.
    pub kind: u8,
    /// Class object id for instances and object arrays; 0 for primitive arrays and class dumps.
    pub class_ref: u64,
    /// HPROF basic type code (4..=11) for primitive arrays; 0 otherwise.
    pub prim_type: u8,
    /// Element width in bytes for arrays; 0 for instances and class dumps.
    pub elem_size: u8,
    /// Element count for arrays; 0 for instances and class dumps.
    pub len: u32,
    /// Length in bytes of the body region that starts at body_offset: the instance
    /// field bytes, len * elem_size for arrays, and the whole class dump body for
    /// KIND_CLASS (starting at aux_offset).
    pub body_len: u32,
    /// File offset of the first field byte (instances) or first element byte (arrays).
    pub body_offset: u64,
    /// File offset of the ClassDump body (instance size, constant pool, static and
    /// instance field descriptors) for KIND_CLASS; 0 otherwise.
    pub aux_offset: u64,
}

/// One GC root located in a heap-dump record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RootRec {
    /// Sub-record tag: 0x01..=0x08, or 0xFF for ROOT_UNKNOWN.
    pub tag: u8,
    /// Target object id (the root's own id for ROOT_THREAD_OBJ).
    pub obj_id: u64,
    /// Thread serial number for the root kinds that carry one.
    pub thread_serial: u32,
    /// Frame index for the root kinds that carry one; u32::MAX when absent.
    pub frame_index: u32,
}

/// One class dump, with its field descriptors borrowed from the walked region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassRec<'a> {
    /// Identity and location of the class dump.
    pub rec: ObjRec,
    /// Superclass object id; 0 for java.lang.Object.
    pub super_id: u64,
    /// Instance size in bytes as recorded in the dump.
    pub instance_size: u32,
    /// Raw static field entries, in file order: repeated (id name, u1 type, value).
    pub static_fields: &'a [u8],
    /// Number of static field entries in static_fields.
    pub static_count: u16,
    /// Raw instance field descriptors, in file order: repeated (id name, u1 type).
    pub instance_fields: &'a [u8],
    /// Number of instance field entries in instance_fields.
    pub instance_count: u16,
}

/// One item reported by walk_heap_region.
pub enum HeapItem<'a> {
    Root(RootRec),
    /// An instance, object array or primitive array.
    Object(ObjRec),
    Class(ClassRec<'a>),
}

/// Where a walked heap-dump record body lives.
///
/// base is added to every reported offset, so a caller walking a window of a file
/// can have the walker report true file offsets; record_tag/record_offset only
/// label error messages.
#[derive(Clone, Copy, Debug)]
pub struct RegionOrigin {
    pub record_tag: u8,
    pub record_offset: u64,
    pub base: u64,
}

impl RegionOrigin {
    /// Origin for a body that starts at file offset base.
    pub fn at(record_tag: u8, record_offset: u64, base: u64) -> Self {
        Self { record_tag, record_offset, base }
    }
}

/// Counts observed while walking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkStats {
    pub instances: u64,
    pub object_arrays: u64,
    pub prim_arrays: u64,
    pub class_dumps: u64,
    pub gc_roots: u64,
}

impl WalkStats {
    fn add(&mut self, other: WalkStats) {
        self.instances += other.instances;
        self.object_arrays += other.object_arrays;
        self.prim_arrays += other.prim_arrays;
        self.class_dumps += other.class_dumps;
        self.gc_roots += other.gc_roots;
    }
}

/// Walk every heap-dump object in data, calling visit once per object in file order.
///
/// id_size is the identifier width from the HPROF header (4 or 8).
///
/// data may be a whole HPROF file (the 1.0.2/1.0.3 header is then skipped, so the
/// reported offsets are true file offsets) or just the record area (offsets are then
/// relative to data).
pub fn walk_objects<F>(data: &[u8], id_size: u8, mut visit: F) -> anyhow::Result<WalkStats>
where
    F: FnMut(&ObjRec),
{
    if id_size != 4 && id_size != 8 {
        bail!("id_size must be 4 or 8, got {id_size}");
    }
    let start = skip_header(data, id_size)?;
    let mut stats = WalkStats::default();
    let mut cur = Cursor { data, pos: start, id_size };
    while cur.pos < data.len() {
        let rec_offset = cur.pos as u64;
        let tag = cur.u8()?;
        let _time_delta = cur.u32()?;
        let body_len = cur.u32()? as usize;
        let body_start = cur.pos;
        let body_end = body_start
            .checked_add(body_len)
            .ok_or_else(|| anyhow!("record tag {tag:#04x} at offset {rec_offset}: body length overflows"))?;
        if body_end > data.len() {
            bail!(
                "record tag {tag:#04x} at offset {rec_offset}: body of {body_len} bytes runs past the end of data ({})",
                data.len()
            );
        }
        if tag == TAG_HEAP_DUMP || tag == TAG_HEAP_DUMP_SEGMENT {
            let origin = RegionOrigin::at(tag, rec_offset, 0);
            let region = walk_heap_region(data, id_size, body_start, body_end, origin, &mut |item| {
                match item {
                    HeapItem::Object(rec) => visit(&rec),
                    HeapItem::Class(cls) => visit(&cls.rec),
                    HeapItem::Root(_) => {}
                }
                Ok(())
            })?;
            stats.add(region);
        }
        cur.pos = body_end;
    }
    Ok(stats)
}

/// Walk the sub-records of one heap-dump record body.
///
/// data[start..end] is the sub-record area of a single HEAP_DUMP /
/// HEAP_DUMP_SEGMENT record. Reported offsets are indices into data unless
/// RegionOrigin::base shifts them into file space, which lets a caller map one
/// record at a time instead of keeping the whole file resident. The sub-record
/// walk may run past end when the layout drifts, so the overshoot is reported
/// instead of being mistaken for a truncated payload.
pub fn walk_heap_region<F>(
    data: &[u8],
    id_size: u8,
    start: usize,
    end: usize,
    origin: RegionOrigin,
    visit: &mut F,
) -> Result<WalkStats>
where
    F: FnMut(HeapItem<'_>) -> Result<()>,
{
    if id_size != 4 && id_size != 8 {
        bail!("id_size must be 4 or 8, got {id_size}");
    }
    if start > end || end > data.len() {
        bail!("heap-dump region {start}..{end} is outside the {} bytes provided", data.len());
    }
    let base = origin.base;
    let mut cur = Cursor { data, pos: start, id_size };
    let mut stats = WalkStats::default();
    while cur.pos < end {
        let sub_offset = cur.pos as u64;
        let sub_tag = cur.u8()?;
        match sub_tag {
            SR_ROOT_UNKNOWN | SR_ROOT_STICKY_CLASS | SR_ROOT_MONITOR_USED => {
                let obj_id = cur.id()?;
                stats.gc_roots += 1;
                visit(HeapItem::Root(RootRec {
                    tag: sub_tag,
                    obj_id,
                    thread_serial: 0,
                    frame_index: u32::MAX,
                }))?;
            }
            SR_ROOT_JNI_GLOBAL => {
                let obj_id = cur.id()?;
                cur.id()?;
                stats.gc_roots += 1;
                visit(HeapItem::Root(RootRec {
                    tag: sub_tag,
                    obj_id,
                    thread_serial: 0,
                    frame_index: u32::MAX,
                }))?;
            }
            SR_ROOT_NATIVE_STACK | SR_ROOT_THREAD_BLOCK => {
                let obj_id = cur.id()?;
                let thread_serial = cur.u32()?;
                stats.gc_roots += 1;
                visit(HeapItem::Root(RootRec {
                    tag: sub_tag,
                    obj_id,
                    thread_serial,
                    frame_index: u32::MAX,
                }))?;
            }
            SR_ROOT_JNI_LOCAL | SR_ROOT_JAVA_FRAME => {
                let obj_id = cur.id()?;
                let thread_serial = cur.u32()?;
                let frame_index = cur.u32()?;
                stats.gc_roots += 1;
                visit(HeapItem::Root(RootRec {
                    tag: sub_tag,
                    obj_id,
                    thread_serial,
                    frame_index,
                }))?;
            }
            SR_ROOT_THREAD_OBJECT => {
                // id thread object, u4 thread serial, u4 stack trace serial.
                let obj_id = cur.id()?;
                let thread_serial = cur.u32()?;
                cur.u32()?;
                stats.gc_roots += 1;
                visit(HeapItem::Root(RootRec {
                    tag: sub_tag,
                    obj_id,
                    thread_serial,
                    frame_index: u32::MAX,
                }))?;
            }
            SR_INSTANCE_DUMP => {
                let addr = cur.id()?;
                let _stack_serial = cur.u32()?;
                let class_ref = cur.id()?;
                let field_bytes = cur.u32()? as u64;
                let body_offset = cur.pos as u64;
                cur.skip(field_bytes)?;
                stats.instances += 1;
                visit(HeapItem::Object(ObjRec {
                    addr,
                    kind: KIND_INSTANCE,
                    class_ref,
                    body_offset: base + body_offset,
                    body_len: field_bytes as u32,
                    ..ObjRec::default()
                }))?;
            }
            SR_OBJ_ARRAY_DUMP => {
                let addr = cur.id()?;
                let _stack_serial = cur.u32()?;
                let len = cur.u32()?;
                let class_ref = cur.id()?;
                let body_offset = cur.pos as u64;
                cur.skip(len as u64 * id_size as u64)?;
                stats.object_arrays += 1;
                visit(HeapItem::Object(ObjRec {
                    addr,
                    kind: KIND_OBJ_ARRAY,
                    class_ref,
                    elem_size: id_size,
                    len,
                    body_offset: base + body_offset,
                    body_len: len.saturating_mul(id_size as u32),
                    ..ObjRec::default()
                }))?;
            }
            SR_PRIM_ARRAY_DUMP => {
                let addr = cur.id()?;
                let _stack_serial = cur.u32()?;
                let len = cur.u32()?;
                let prim_type = cur.u8()?;
                let elem_size = field_width(prim_type, id_size, base + cur.pos as u64)?;
                let body_offset = cur.pos as u64;
                cur.skip(len as u64 * elem_size)?;
                stats.prim_arrays += 1;
                visit(HeapItem::Object(ObjRec {
                    addr,
                    kind: KIND_PRIM_ARRAY,
                    prim_type,
                    elem_size: elem_size as u8,
                    len,
                    body_offset: base + body_offset,
                    body_len: len.saturating_mul(elem_size as u32),
                    ..ObjRec::default()
                }))?;
            }
            SR_CLASS_DUMP => {
                // Everything after the tag byte is the class dump body.
                let aux_start = cur.pos;
                let class_id = cur.id()?;
                let _stack_serial = cur.u32()?;
                let super_id = cur.id()?;
                // class loader, signers, protection domain and two reserved ids.
                cur.skip(5 * id_size as u64)?;
                let instance_size = cur.u32()?;

                let constant_pool_count = cur.u16()?;
                for _ in 0..constant_pool_count {
                    cur.skip(2)?;
                    let ty = cur.u8()?;
                    let width = field_width(ty, id_size, base + cur.pos as u64)?;
                    cur.skip(width)?;
                }

                let static_count = cur.u16()?;
                let static_start = cur.pos;
                for _ in 0..static_count {
                    cur.id()?;
                    let ty = cur.u8()?;
                    let width = field_width(ty, id_size, base + cur.pos as u64)?;
                    cur.skip(width)?;
                }
                let static_end = cur.pos;

                let instance_count = cur.u16()?;
                let instance_start = cur.pos;
                for _ in 0..instance_count {
                    cur.id()?;
                    cur.u8()?;
                }
                let instance_end = cur.pos;

                stats.class_dumps += 1;
                visit(HeapItem::Class(ClassRec {
                    rec: ObjRec {
                        addr: class_id,
                        kind: KIND_CLASS,
                        aux_offset: base + aux_start as u64,
                        body_len: (instance_end - aux_start).min(u32::MAX as usize) as u32,
                        ..ObjRec::default()
                    },
                    super_id,
                    instance_size,
                    static_fields: &data[static_start..static_end],
                    static_count,
                    instance_fields: &data[instance_start..instance_end],
                    instance_count,
                }))?;
            }
            other => bail!(
                "unknown heap-dump sub-record tag {other:#04x} at offset {} (in record tag {:#04x} at offset {})",
                base + sub_offset,
                origin.record_tag,
                origin.record_offset
            ),
        }

        if cur.pos as u64 > end as u64 {
            bail!(
                "heap-dump record tag {:#04x} at offset {}: sub-record tag {sub_tag:#04x} at offset {} overshoots body end {} by {} bytes (layout drift)",
                origin.record_tag,
                origin.record_offset,
                base + sub_offset,
                base + end as u64,
                cur.pos as u64 - end as u64
            );
        }
    }
    debug_assert_eq!(cur.pos, end, "heap-dump body must be consumed exactly");
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Field descriptors
// ---------------------------------------------------------------------------

/// Cursor over the raw field-descriptor bytes of a ClassRec.
pub struct Fields<'a> {
    data: &'a [u8],
    pos: usize,
    id_size: u8,
}

impl<'a> Fields<'a> {
    /// Read data as id_size-wide descriptors.
    pub fn new(data: &'a [u8], id_size: u8) -> Self {
        Self { data, pos: 0, id_size }
    }

    /// Next static field as (name string id, type code), skipping its value bytes.
    pub fn next_static(&mut self) -> Option<(u64, u8)> {
        let name = self.id()?;
        let ty = *self.data.get(self.pos)?;
        self.pos += 1;
        self.pos += basic_width(ty, self.id_size)?;
        if self.pos > self.data.len() {
            return None;
        }
        Some((name, ty))
    }

    /// Next instance field as (name string id, type code).
    pub fn next_instance(&mut self) -> Option<(u64, u8)> {
        let name = self.id()?;
        let ty = *self.data.get(self.pos)?;
        self.pos += 1;
        Some((name, ty))
    }

    fn id(&mut self) -> Option<u64> {
        let w = self.id_size as usize;
        let end = self.pos.checked_add(w)?;
        let raw = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(if w == 4 {
            u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as u64
        } else {
            let mut b = [0u8; 8];
            b.copy_from_slice(raw);
            u64::from_be_bytes(b)
        })
    }
}

// ---------------------------------------------------------------------------
// Record / sub-record tags
// ---------------------------------------------------------------------------

const SR_ROOT_JNI_GLOBAL: u8 = 0x01;
const SR_ROOT_JNI_LOCAL: u8 = 0x02;
const SR_ROOT_JAVA_FRAME: u8 = 0x03;
const SR_ROOT_NATIVE_STACK: u8 = 0x04;
const SR_ROOT_STICKY_CLASS: u8 = 0x05;
const SR_ROOT_THREAD_BLOCK: u8 = 0x06;
const SR_ROOT_MONITOR_USED: u8 = 0x07;
const SR_ROOT_THREAD_OBJECT: u8 = 0x08;
const SR_CLASS_DUMP: u8 = 0x20;
const SR_INSTANCE_DUMP: u8 = 0x21;
const SR_OBJ_ARRAY_DUMP: u8 = 0x22;
const SR_PRIM_ARRAY_DUMP: u8 = 0x23;
const SR_ROOT_UNKNOWN: u8 = 0xFF;

const MAGIC: &[u8] = b"JAVA PROFILE 1.0.";

// ---------------------------------------------------------------------------
// Bounds-checked cursor
// ---------------------------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    id_size: u8,
}

impl<'a> Cursor<'a> {
    fn need(&self, n: u64) -> Result<()> {
        let end = (self.pos as u64)
            .checked_add(n)
            .ok_or_else(|| anyhow!("offset {}: payload length {n} overflows", self.pos))?;
        if end > self.data.len() as u64 {
            bail!(
                "offset {}: truncated payload, need {n} bytes but only {} remain",
                self.pos,
                (self.data.len() as u64).saturating_sub(self.pos as u64)
            );
        }
        Ok(())
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        self.need(n)?;
        self.pos += n as usize;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16> {
        self.need(2)?;
        let p = self.pos;
        let v = u16::from_be_bytes([self.data[p], self.data[p + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        let p = self.pos;
        let v = u32::from_be_bytes([self.data[p], self.data[p + 1], self.data[p + 2], self.data[p + 3]]);
        self.pos += 4;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        let p = self.pos;
        let mut b = [0u8; 8];
        b.copy_from_slice(&self.data[p..p + 8]);
        self.pos += 8;
        Ok(u64::from_be_bytes(b))
    }

    fn id(&mut self) -> Result<u64> {
        if self.id_size == 4 {
            Ok(self.u32()? as u64)
        } else {
            self.u64()
        }
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Offset of the first top-level record, skipping the header when data carries one.
fn skip_header(data: &[u8], id_size: u8) -> Result<usize> {
    if !data.starts_with(MAGIC) {
        return Ok(0);
    }
    let version = match data.get(MAGIC.len()) {
        Some(&v) => v,
        None => bail!("truncated HPROF header: no version byte"),
    };
    if version != b'2' && version != b'3' {
        bail!(
            "unsupported HPROF version 1.0.{}: this record layout is defined for 1.0.2 and 1.0.3",
            version as char
        );
    }
    let mut cur = Cursor {
        data,
        pos: MAGIC.len() + 1,
        id_size,
    };
    if cur.u8()? != 0 {
        bail!("HPROF header: expected NUL terminator after the version string");
    }
    let header_id_size = cur.u32()?;
    if header_id_size != 4 && header_id_size != 8 {
        bail!("HPROF header id_size {header_id_size} is not 4 or 8");
    }
    if header_id_size != id_size as u32 {
        bail!("id_size {id_size} does not match the HPROF header id_size {header_id_size}");
    }
    let _timestamp_millis = cur.u64()?;
    Ok(cur.pos)
}

// ---------------------------------------------------------------------------
// Field widths
// ---------------------------------------------------------------------------

/// Width in bytes of an HPROF basic type, or None when the code is not one.
fn basic_width(ty: u8, id_size: u8) -> Option<usize> {
    Some(match ty {
        2 => id_size as usize,
        4 | 8 => 1,
        5 | 9 => 2,
        6 | 10 => 4,
        7 | 11 => 8,
        _ => return None,
    })
}

/// Width in bytes of an HPROF basic type; at is only used for the error message.
fn field_width(ty: u8, id_size: u8, at: u64) -> Result<u64> {
    match basic_width(ty, id_size) {
        Some(w) => Ok(w as u64),
        None => bail!("unknown HPROF field type {ty} at offset {at}"),
    }
}
