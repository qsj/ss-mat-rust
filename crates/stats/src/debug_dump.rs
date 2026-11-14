//! Opt-in forensic dumps controlled by `SS_MAT_DEBUG_DIR`.
//!
//! Everything in this module is inert unless the environment variable is set
//! before the process starts. The regular analysis path never reads these
//! structures, so report numbers are unaffected when the switch is off.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::duplicate_strings::DupStringGroup;
use crate::engine::StatsEngine;

/// Return the dump directory when `SS_MAT_DEBUG_DIR` is set to a non-empty value.
pub fn debug_dir() -> Option<PathBuf> {
    std::env::var("SS_MAT_DEBUG_DIR")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(PathBuf::from)
}

pub fn enabled() -> bool {
    debug_dir().is_some()
}

/// One primitive array that belongs to a confirmed duplicate group.
#[derive(Clone)]
pub struct DupPrimMemberDebug {
    pub obj_id: u64,
    pub string_backing: bool,
    pub container_backing: bool,
}

/// One confirmed duplicate group as seen by `detect_duplicate_prim_arrays`.
#[derive(Clone)]
pub struct DupPrimGroupDebug {
    pub class_name: String,
    pub array_length: i32,
    pub element_size: i32,
    /// Array body size used by the detector for waste accounting
    /// (`array_shallow_size` in `linear_scan`, i.e. align(header + data)).
    pub body_size: i64,
    pub count: usize,
    pub wasted_bytes: i64,
    /// FNV-1a 64-bit hash of the array body (stable across runs/versions).
    pub content_hash: u64,
    pub members: Vec<DupPrimMemberDebug>,
}

/// Aggregate counters captured by the detector, independent of group contents.
#[derive(Clone, Copy, Default)]
pub struct DupPrimScanTotals {
    pub scanned_arrays: usize,
    pub eligible_arrays: usize,
    pub arrays_in_groups: usize,
    pub groups: usize,
    pub waste_bytes: i64,
    /// Java `DuplicateObjectsReport.totalGroupCount` (enumerated first-bucket
    /// nodes + both pending map sizes).
    pub unique_count: usize,
}

static DUP_PRIM_GROUPS: OnceLock<Mutex<Vec<DupPrimGroupDebug>>> = OnceLock::new();
static DUP_PRIM_TOTALS: OnceLock<Mutex<DupPrimScanTotals>> = OnceLock::new();

fn group_store() -> &'static Mutex<Vec<DupPrimGroupDebug>> {
    DUP_PRIM_GROUPS.get_or_init(|| Mutex::new(Vec::new()))
}

fn totals_store() -> &'static Mutex<DupPrimScanTotals> {
    DUP_PRIM_TOTALS.get_or_init(|| Mutex::new(DupPrimScanTotals::default()))
}

/// Store duplicate-group forensics. No-op unless the debug switch is on.
pub fn record_dup_prim_groups(groups: Vec<DupPrimGroupDebug>) {
    if groups.is_empty() {
        return;
    }
    group_store().lock().unwrap().extend(groups);
}

pub fn record_dup_prim_totals(totals: DupPrimScanTotals) {
    *totals_store().lock().unwrap() = totals;
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Numbers emitted to `s14_summary.txt` and echoed on stderr.
pub struct S14DumpSummary {
    pub groups: usize,
    pub arrays_in_groups: usize,
    pub waste_bytes: i64,
    pub live_members: usize,
    pub garbage_members: usize,
    pub string_backing_members: usize,
    pub container_backing_members: usize,
    pub groups_with_garbage: usize,
    pub groups_all_garbage: usize,
    pub groups_all_live: usize,
}

/// Write `s14_groups.tsv`, `s14_members.tsv` and `s14_summary.txt`.
///
/// `live_prim_obj_ids` must contain the addresses of primitive arrays that the
/// BFS marked reachable. Arrays missing from that set are reported as garbage.
pub fn write_s14_bundle(dir: &Path, live_prim_obj_ids: &HashSet<u64>) -> std::io::Result<S14DumpSummary> {
    std::fs::create_dir_all(dir)?;

    let mut groups = group_store().lock().unwrap().clone();
    // Deterministic order: biggest waste first, then class/len/hash.
    groups.sort_by(|a, b| {
        b.wasted_bytes
            .cmp(&a.wasted_bytes)
            .then_with(|| a.class_name.cmp(&b.class_name))
            .then_with(|| a.array_length.cmp(&b.array_length))
            .then_with(|| a.content_hash.cmp(&b.content_hash))
    });

    let mut summary = S14DumpSummary {
        groups: groups.len(),
        arrays_in_groups: 0,
        waste_bytes: 0,
        live_members: 0,
        garbage_members: 0,
        string_backing_members: 0,
        container_backing_members: 0,
        groups_with_garbage: 0,
        groups_all_garbage: 0,
        groups_all_live: 0,
    };

    for g in &groups {
        summary.arrays_in_groups += g.count;
        summary.waste_bytes += g.wasted_bytes;
        let mut live = 0usize;
        for m in &g.members {
            if live_prim_obj_ids.contains(&m.obj_id) {
                live += 1;
            }
            if m.string_backing {
                summary.string_backing_members += 1;
            }
            if m.container_backing {
                summary.container_backing_members += 1;
            }
        }
        summary.live_members += live;
        summary.garbage_members += g.count - live;
        if live == g.count {
            summary.groups_all_live += 1;
        } else if live == 0 {
            summary.groups_all_garbage += 1;
        } else {
            summary.groups_with_garbage += 1;
        }
    }

    let groups_path = dir.join("s14_groups.tsv");
    let members_path = dir.join("s14_members.tsv");
    let summary_path = dir.join("s14_summary.txt");

    {
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&groups_path)?);
        writeln!(
            out,
            "group_id\tclass_name\tarray_length\telement_size\tbody_size\tcount\twaste_bytes\tcontent_hash\tmember_live\tmember_garbage\tmember_string_backing\tmember_container_backing\trep_obj_id_hex\trep_obj_id_dec\tdetector_excludes_string_backing\tdetector_excludes_container_backing"
        )?;
        for (i, g) in groups.iter().enumerate() {
            let group_id = i + 1;
            let live = g
                .members
                .iter()
                .filter(|m| live_prim_obj_ids.contains(&m.obj_id))
                .count();
            let string_backing = g.members.iter().filter(|m| m.string_backing).count();
            let container_backing = g.members.iter().filter(|m| m.container_backing).count();
            let rep = g.members.iter().map(|m| m.obj_id).min().unwrap_or(0);
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:016x}\t{}\t{}\t{}\t{}\t0x{:x}\t{}\t0\t0",
                group_id,
                g.class_name,
                g.array_length,
                g.element_size,
                g.body_size,
                g.count,
                g.wasted_bytes,
                g.content_hash,
                live,
                g.count - live,
                string_backing,
                container_backing,
                rep,
                rep,
            )?;
        }
    }

    {
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&members_path)?);
        writeln!(
            out,
            "group_id\tobj_id_hex\tobj_id_dec\tlive\tstring_backing\tcontainer_backing\tclass_name\tarray_length\tbody_size\tcontent_hash"
        )?;
        for (i, g) in groups.iter().enumerate() {
            let group_id = i + 1;
            let mut members: Vec<&DupPrimMemberDebug> = g.members.iter().collect();
            members.sort_by_key(|m| m.obj_id);
            for m in members {
                writeln!(
                    out,
                    "{}\t0x{:x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:016x}",
                    group_id,
                    m.obj_id,
                    m.obj_id,
                    if live_prim_obj_ids.contains(&m.obj_id) { 1 } else { 0 },
                    if m.string_backing { 1 } else { 0 },
                    if m.container_backing { 1 } else { 0 },
                    g.class_name,
                    g.array_length,
                    g.body_size,
                    g.content_hash,
                )?;
            }
        }
    }

    let totals = *totals_store().lock().unwrap();
    let mut text = String::new();
    let _ = writeln!(text, "SS-Mat Rust S14 (duplicate primitive arrays) forensic dump");
    let _ = writeln!(text, "generated: {}", timestamp());
    let _ = writeln!(text, "detector: linear_scan::detect_duplicate_prim_arrays");
    let _ = writeln!(
        text,
        "detector_semantics: Java PrimitiveArraySizeAccumulator (pass2 non-char + pass1 char); \
         Java hashByteArray; merge enumerates only the hash-0 groupMap bucket; \
         NO live/garbage filter"
    );
    let _ = writeln!(
        text,
        "body_size_formula: align(header_size + 4 + length * element_size, 8) \
         [linear_scan::array_shallow_size, Java PrimitiveArrayInstance.getBodySize]"
    );
    let _ = writeln!(text);
    let _ = writeln!(text, "scanned_prim_arrays\t{}", totals.scanned_arrays);
    let _ = writeln!(text, "eligible_arrays\t{}", totals.eligible_arrays);
    let _ = writeln!(text, "duplicate_groups\t{}", totals.groups);
    let _ = writeln!(text, "unique_count_field2\t{}", totals.unique_count);
    let _ = writeln!(text, "arrays_in_duplicate_groups\t{}", totals.arrays_in_groups);
    let _ = writeln!(text, "waste_bytes\t{}", totals.waste_bytes);
    let _ = writeln!(text, "waste_kb\t{:.0}", totals.waste_bytes as f64 / 1024.0);
    let _ = writeln!(text);
    let _ = writeln!(text, "live_members\t{}", summary.live_members);
    let _ = writeln!(text, "garbage_members\t{}", summary.garbage_members);
    let _ = writeln!(text, "members_that_are_string_backing\t{}", summary.string_backing_members);
    let _ = writeln!(
        text,
        "members_that_are_container_backing_partial\t{}",
        summary.container_backing_members
    );
    let _ = writeln!(text, "groups_all_live\t{}", summary.groups_all_live);
    let _ = writeln!(text, "groups_with_garbage\t{}", summary.groups_with_garbage);
    let _ = writeln!(text, "groups_all_garbage\t{}", summary.groups_all_garbage);
    let _ = writeln!(text);
    let _ = writeln!(text, "# Per-class breakdown (groups, arrays, waste_bytes)");
    let mut per_class: std::collections::HashMap<&str, (usize, usize, i64)> =
        std::collections::HashMap::new();
    for g in &groups {
        let e = per_class.entry(g.class_name.as_str()).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += g.count;
        e.2 += g.wasted_bytes;
    }
    let mut rows: Vec<_> = per_class.into_iter().collect();
    rows.sort_by(|a, b| b.1 .2.cmp(&a.1 .2));
    for (name, (groups_n, arrays_n, waste)) in rows {
        let _ = writeln!(
            text,
            "per_class\t{}\tgroups={}\tarrays={}\twaste_bytes={}",
            name, groups_n, arrays_n, waste
        );
    }
    let _ = writeln!(text);
    let _ = writeln!(
        text,
        "# Note: the report's 'Duplicate groups' column uses per_class.len() \
         (primitive array TYPES with hash-0 waste), while totalGroupCount \
         (report 'Unique arrays') = enumerated hash-0 nodes + both pending maps. \
         'Total arrays' is all primitive arrays scanned."
    );

    std::fs::write(&summary_path, text)?;
    Ok(summary)
}

fn timestamp() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => format!("unix={}s", d.as_secs()),
        Err(_) => "unknown".to_string(),
    }
}

/// Build a primitive-array liveness set from the persisted BFS traversal data.
/// Only kind==1 (primitive array) objects are considered, keeping memory small.
pub fn live_primitive_array_ids(
    traversal: &crate::traversal_data::TraversalData,
) -> HashSet<u64> {
    traversal
        .objects
        .iter()
        .enumerate()
        .filter(|(i, o)| o.kind == 1 && traversal.visited[*i])
        .map(|(_, o)| o.obj_id)
        .collect()
}

// ---------------------------------------------------------------------------
// Aggregate problem forensics (S8/S9/S10/S11/S12 totals per class)
// ---------------------------------------------------------------------------

static DUP_STRING_GROUPS: OnceLock<Mutex<Vec<DupStringGroup>>> = OnceLock::new();

fn string_group_store() -> &'static Mutex<Vec<DupStringGroup>> {
    DUP_STRING_GROUPS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Retain every duplicate-string group (the report itself keeps only the top
/// 200). No-op unless the debug switch is on; called from
/// `StringDupCollector::set_results`.
pub fn record_dup_string_groups(groups: Vec<DupStringGroup>) {
    if groups.is_empty() {
        return;
    }
    *string_group_store().lock().unwrap() = groups;
}

/// Java `CollectionProblemKind` names mirrored for the dump.
fn java_kind_name(kind: u8) -> &'static str {
    crate::problem_types::collection_problem_name(kind)
}

/// Rust's primitive-array synthetic class name -> Java's HPROF class name.
fn synthetic_to_hprof(name: &str) -> Option<&'static str> {
    match name {
        "boolean[]" => Some("[Z"),
        "byte[]" => Some("[B"),
        "char[]" => Some("[C"),
        "short[]" => Some("[S"),
        "int[]" => Some("[I"),
        "long[]" => Some("[J"),
        "float[]" => Some("[F"),
        "double[]" => Some("[D"),
        _ => None,
    }
}

fn prim_char_to_name(c: char) -> Option<&'static str> {
    match c {
        'Z' => Some("boolean"),
        'B' => Some("byte"),
        'C' => Some("char"),
        'S' => Some("short"),
        'I' => Some("int"),
        'J' => Some("long"),
        'F' => Some("float"),
        'D' => Some("double"),
        _ => None,
    }
}

/// Java `JavaClassInfo.isPrimitiveArray()`: arrayDimensions == 1 and a
/// primitive element type (HPROF name `[B`, not the synthetic `byte[]`).
fn is_single_dim_prim_hprof(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() == 2 && bytes[0] == b'[' && prim_char_to_name(bytes[1] as char).is_some()
}

/// HPROF name with >=2 '[' dimensions whose element is a primitive type.
/// Java treats these as neither object arrays nor primitive arrays
/// (`isPrimitiveArrayType` is true but `arrayDimensions > 1`).
fn is_multi_dim_prim_hprof(name: &str) -> bool {
    if !name.starts_with("[[") {
        return false;
    }
    let rest = name.trim_start_matches('[');
    rest.len() == 1 && prim_char_to_name(rest.chars().next().unwrap()).is_some()
}

/// Mirrors `linear_scan::is_collection_class_name`, the set of collections the
/// Rust scan currently classifies (Java's ContainerRegistry is broader; see
/// README).
fn is_collection_class_name(name: &str) -> bool {
    const NAMES: &[&str] = &[
        "java/util/HashMap", "java/util/LinkedHashMap", "java/util/Hashtable",
        "java/util/concurrent/ConcurrentHashMap",
        "java/util/TreeMap",
        "java/util/EnumMap",
        "java/util/ArrayList", "java/util/LinkedList", "java/util/Vector",
        "java/util/HashSet", "java/util/LinkedHashSet", "java/util/TreeSet",
        "java/util/concurrent/CopyOnWriteArrayList",
        "java/util/concurrent/ConcurrentSkipListMap",
        "java/util/concurrent/ConcurrentSkipListSet",
    ];
    NAMES.iter().any(|n| name == *n)
}

/// Java's `PerClassProblemStats.problemObjectCount`: 1=collection,
/// 2=object_array, 3=primitive_array, 0=other.
fn category_of(class_name: &str) -> (u8, &'static str) {
    if synthetic_to_hprof(class_name).is_some() || is_single_dim_prim_hprof(class_name) {
        (3, "primitive_array")
    } else if is_multi_dim_prim_hprof(class_name) {
        // Java leaves 2-D primitive arrays with plain PerClassStats.
        (0, "other")
    } else if class_name.starts_with('[') || class_name.ends_with("[]") {
        (2, "object_array")
    } else if is_collection_class_name(class_name) {
        (1, "collection")
    } else {
        (0, "other")
    }
}

/// Java-visible class name: synthetic Rust primitive-array classes become
/// their HPROF name, and '/' becomes '.' like Java's JavaClassInfo.name.
fn java_class_name(raw: &str) -> String {
    match synthetic_to_hprof(raw) {
        Some(h) => h.to_string(),
        None => raw.replace('/', "."),
    }
}

/// Map the Rust `BadArrayRecord.kind` (0..4) to Java's CollectionProblemKind.
fn object_array_java_kind(kind: u8) -> u8 {
    use crate::problem_types::*;
    match kind {
        0 => COLLECTION_LEN_ZERO,
        1 => COLLECTION_LEN_ONE,
        2 => COLLECTION_EMPTY,
        3 => COLLECTION_SINGLE_ELEM,
        4 => COLLECTION_SPARSE,
        _ => COLLECTION_KIND_NONE,
    }
}

/// Map the Rust `BadPrimArrayRecord.kind` (0..5) to Java's CollectionProblemKind.
fn prim_array_java_kind(kind: u8) -> u8 {
    use crate::problem_types::*;
    match kind {
        0 => COLLECTION_LEN_ZERO,
        1 => COLLECTION_LEN_ONE,
        2 => COLLECTION_EMPTY,
        3 => COLLECTION_TRAIL_ZEROES,
        4 => COLLECTION_SPARSE,
        5 => COLLECTION_ZERO_HIGH_BITS,
        _ => COLLECTION_KIND_NONE,
    }
}

#[derive(Clone)]
struct ClassProblemAgg {
    kind_count: [u64; crate::problem_types::COLLECTION_KIND_COUNT],
    kind_size: [i64; crate::problem_types::COLLECTION_KIND_COUNT],
    bad_count: u64,
    bad_size: i64,
    instance_count: u64,
    array_count: u64,
}

impl Default for ClassProblemAgg {
    fn default() -> Self {
        Self {
            kind_count: [0; crate::problem_types::COLLECTION_KIND_COUNT],
            kind_size: [0; crate::problem_types::COLLECTION_KIND_COUNT],
            bad_count: 0,
            bad_size: 0,
            instance_count: 0,
            array_count: 0,
        }
    }
}

/// Numbers echoed on stderr after `write_problem_bundle`.
pub struct ProblemDumpSummary {
    pub s8_total_strings: usize,
    pub s8_unique_strings: usize,
    pub s8_duplicate_values: usize,
    pub s8_wasted_bytes: i64,
    pub s8_groups_dumped: usize,
    pub collection_total_objects: u64,
    pub collection_problem_count: u64,
    pub collection_overhead: i64,
    pub obj_array_total_objects: u64,
    pub obj_array_problem_count: u64,
    pub obj_array_overhead: i64,
    pub prim_array_total_objects: u64,
    pub prim_array_problem_count: u64,
    pub prim_array_overhead: i64,
}

fn escape_preview(content: &str, max_chars: usize) -> String {
    let mut out = String::new();
    let mut taken = 0usize;
    let mut truncated = false;
    for ch in content.chars() {
        if taken >= max_chars {
            truncated = true;
            break;
        }
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(ch),
        }
        taken += 1;
    }
    if truncated {
        out.push_str("...");
    }
    out
}

/// Java `PrimitiveTypeMapper.abbreviateClassName`.
fn abbreviate_class_name(name: &str) -> String {
    if name == "java.lang.String" {
        return "String".to_string();
    }
    if name == "java.lang.Object" {
        return "Object".to_string();
    }
    let last_dot = match name.rfind('.') {
        Some(p) => p,
        None => return name.to_string(),
    };
    let pkg = &name[..=last_dot];
    let abbrev = match pkg {
        "java.lang." => "j.l.",
        "java.util." => "j.u.",
        "java.util.concurrent." => "j.u.concurrent.",
        "java.lang.ref." => "j.l.r.",
        _ => pkg,
    };
    format!("{}{}", abbrev, &name[last_dot + 1..])
}

/// Java `JavaClassInfo.displayName` for the class shapes the dump carries
/// (regular classes, object arrays and 1-D primitive arrays).
fn java_display_name(raw: &str) -> String {
    if raw.starts_with('[') {
        let mut dims = 0usize;
        for ch in raw.chars() {
            if ch == '[' {
                dims += 1;
            } else {
                break;
            }
        }
        let rest = &raw[dims..];
        if rest.len() == 1 {
            if let Some(type_name) = prim_char_to_name(rest.chars().next().unwrap()) {
                let mut out = type_name.to_string();
                for _ in 0..dims {
                    out.push_str("[]");
                }
                return out;
            }
        }
        let element = if raw.ends_with(';') && raw.len() > dims + 1 {
            &raw[dims + 1..raw.len() - 1]
        } else {
            rest
        };
        let mut out = abbreviate_class_name(&element.replace('/', "."));
        for _ in 0..dims {
            out.push_str("[]");
        }
        return out;
    }
    if raw.ends_with("[]") {
        return raw.to_string();
    }
    abbreviate_class_name(&raw.replace('/', "."))
}

/// Write `problem_stats.tsv`, `s8_summary.tsv`, `s8_groups.tsv` and
/// `boxed_summary.tsv`. Inert unless `SS_MAT_DEBUG_DIR` is set; never called
/// from the normal report path.
pub fn write_problem_bundle(
    dir: &Path,
    stats: &StatsEngine,
) -> std::io::Result<ProblemDumpSummary> {
    use std::sync::atomic::Ordering::Relaxed;
    use crate::problem_types::*;

    std::fs::create_dir_all(dir)?;

    // ---- Aggregate per-class problem records (Java ordinals) ----
    // Rust keeps synthetic 1-D primitive-array classes ("byte[]") next to the
    // HPROF-named ones ("[B"); Java only has the latter. Merge the aliases so
    // one row per Java class is emitted.
    let n = stats.per_class.len();
    let mut hprof_idx: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, pc) in stats.per_class.iter().enumerate() {
        if is_single_dim_prim_hprof(&pc.class_name) {
            hprof_idx.insert(pc.class_name.as_str(), i);
        }
    }
    let mut canonical_of: Vec<usize> = (0..n).collect();
    for (i, pc) in stats.per_class.iter().enumerate() {
        if let Some(h) = synthetic_to_hprof(&pc.class_name) {
            if let Some(&j) = hprof_idx.get(h) {
                canonical_of[i] = j;
            }
        }
    }

    let mut aggs: Vec<ClassProblemAgg> = vec![ClassProblemAgg::default(); n];
    for (i, pc) in stats.per_class.iter().enumerate() {
        let c = canonical_of[i];
        aggs[c].instance_count += pc.instance_count.load(Relaxed);
        aggs[c].array_count += pc.array_live_count.load(Relaxed) as u64;
    }

    {
        let records = stats.bad_collection_records.lock().unwrap();
        for r in records.iter() {
            if r.kind == COLLECTION_KIND_NONE {
                continue;
            }
            let idx = r.class_idx as usize;
            if idx >= aggs.len() {
                continue;
            }
            let idx = canonical_of[idx];
            let k = r.kind as usize;
            aggs[idx].kind_count[k] += 1;
            aggs[idx].kind_size[k] += r.waste;
            aggs[idx].bad_count += 1;
            aggs[idx].bad_size += r.waste;
        }
    }

    // Java never runs processObjectArray() for container backing arrays; the
    // scan does, so exclude them here exactly like the report does.
    let container_backing: HashSet<u64> = stats
        .bad_collection_records
        .lock()
        .unwrap()
        .iter()
        .map(|r| r.backing_array_id)
        .filter(|id| *id != 0)
        .collect();

    {
        let records = stats.bad_array_records.lock().unwrap();
        for r in records.iter() {
            if container_backing.contains(&r.obj_id) {
                continue;
            }
            let idx = r.class_idx as usize;
            if idx >= aggs.len() {
                continue;
            }
            let idx = canonical_of[idx];
            let k = object_array_java_kind(r.kind) as usize;
            aggs[idx].kind_count[k] += 1;
            aggs[idx].kind_size[k] += r.size;
            aggs[idx].bad_count += 1;
            aggs[idx].bad_size += r.size;
        }
    }

    {
        let records = stats.bad_prim_array_records.lock().unwrap();
        for r in records.iter() {
            let idx = r.class_idx as usize;
            if idx >= aggs.len() {
                continue;
            }
            let idx = canonical_of[idx];
            let k = prim_array_java_kind(r.kind) as usize;
            aggs[idx].kind_count[k] += 1;
            aggs[idx].kind_size[k] += r.waste;
            aggs[idx].bad_count += 1;
            aggs[idx].bad_size += r.waste;
        }
    }

    // ---- problem_stats.tsv (schema mirrors Java DebugDumper) ----
    // Java sorts by JavaClassInfo.name (dotted). Rust class names keep the
    // HPROF '/', so sort on the dotted form for identical row order. Synthetic
    // primitive-array aliases are folded into their HPROF class row.
    let mut rows: Vec<(String, usize)> = stats
        .per_class
        .iter()
        .enumerate()
        .filter(|(i, _)| canonical_of[*i] == *i)
        .map(|(i, pc)| (java_class_name(&pc.class_name), i))
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    {
        let mut out = BufWriter::with_capacity(1 << 20, File::create(dir.join("problem_stats.tsv"))?);
        write!(
            out,
            "class_name\tdisplay_name\tproblem_object_count\tinstance_count\tproblem_array_count"
        )?;
        for k in 0..COLLECTION_KIND_COUNT {
            write!(out, "\tproblem_kind_count_{}", collection_problem_enum_name(k as u8))?;
        }
        for k in 0..COLLECTION_KIND_COUNT {
            write!(out, "\tproblem_kind_size_{}", collection_problem_enum_name(k as u8))?;
        }
        writeln!(out)?;

        for (class_name, i) in &rows {
            let pc = &stats.per_class[*i];
            let (cat_id, _) = category_of(&pc.class_name);
            // Java's PerClassStats.instanceCount includes arrays for array
            // classes (HprofDumpReader increments it in addObjectArray /
            // addPrimitiveArray); Rust keeps them in array_live_count.
            let instance_count = aggs[*i].instance_count + aggs[*i].array_count;
            write!(
                out,
                "{}\t{}\t{}\t{}\t{}",
                class_name,
                java_display_name(&pc.class_name),
                cat_id,
                instance_count,
                aggs[*i].bad_count,
            )?;
            for k in 0..COLLECTION_KIND_COUNT {
                write!(out, "\t{}", aggs[*i].kind_count[k])?;
            }
            for k in 0..COLLECTION_KIND_COUNT {
                write!(out, "\t{}", aggs[*i].kind_size[k])?;
            }
            writeln!(out)?;
        }
    }

    // ---- s8_summary.tsv / s8_groups.tsv ----
    let report = stats.string_dup.get_report();
    let mut groups = string_group_store().lock().unwrap().clone();
    groups.sort_by(|a, b| {
        b.wasted_bytes
            .cmp(&a.wasted_bytes)
            .then_with(|| fnv1a64(a.content.as_bytes()).cmp(&fnv1a64(b.content.as_bytes())))
            .then_with(|| b.count.cmp(&a.count))
    });

    {
        // Java schema: RefChainGroupReport(uniqueStringCount=total strings,
        // duplicateStringCount=unique strings, getGroupCount()=duplicate
        // values, wastedBytes). Field names are Java's (misleading) names.
        let mut out = BufWriter::with_capacity(1 << 16, File::create(dir.join("s8_summary.tsv"))?);
        writeln!(
            out,
            "unique_string_count\tduplicate_string_count\tgroup_count\twasted_bytes"
        )?;
        writeln!(
            out,
            "{}\t{}\t{}\t{}",
            report.total_strings, report.unique_strings, report.duplicate_group_count, report.wasted_bytes
        )?;
    }

    {
        let mut out = BufWriter::with_capacity(1 << 20, File::create(dir.join("s8_groups.tsv"))?);
        writeln!(
            out,
            "group_index\trepresentative_index\tobject_size\tunique_value_count\tduplicate_count\ttotal_waste\tgroup_id\trepresentative_object_group_id\trepresentative_class\trepresentative_body_size\trep_obj_id_hex\tcontent_hash\tcontent_len_chars\tcontent_preview\tsample_obj_ids_hex"
        )?;
        for (i, g) in groups.iter().enumerate() {
            let rep = g.sample_obj_ids.iter().copied().min().unwrap_or(0);
            let samples = g
                .sample_obj_ids
                .iter()
                .map(|id| format!("0x{:x}", id))
                .collect::<Vec<_>>()
                .join(",");
            // Java's DuplicateGroupEntry names are scrambled: representative_index
            // holds the group id, object_size holds the representative object
            // index. Rust has no object index, so object_size carries the HPROF
            // address (decimal).
            let body_size = if report.string_body_size > 0 {
                report.string_body_size.to_string()
            } else {
                "-".to_string()
            };
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t-\tString\t{}\t0x{:x}\t{:016x}\t{}\t{}\t{}",
                i,
                i,
                rep,
                g.backing_array_count,
                g.count,
                g.wasted_bytes,
                i,
                body_size,
                rep,
                fnv1a64(g.content.as_bytes()),
                g.content.chars().count(),
                escape_preview(&g.content, 120),
                samples,
            )?;
        }
    }

    // ---- boxed_summary.tsv (schema mirrors Java DebugDumper) ----
    let mut summary = ProblemDumpSummary {
        s8_total_strings: report.total_strings,
        s8_unique_strings: report.unique_strings,
        s8_duplicate_values: report.duplicate_group_count,
        s8_wasted_bytes: report.wasted_bytes,
        s8_groups_dumped: groups.len(),
        collection_total_objects: 0,
        collection_problem_count: 0,
        collection_overhead: 0,
        obj_array_total_objects: 0,
        obj_array_problem_count: 0,
        obj_array_overhead: 0,
        prim_array_total_objects: 0,
        prim_array_problem_count: 0,
        prim_array_overhead: 0,
    };

    // Java BoxedNumberOverheadSummary: for every class with
    // problemObjectCount == source, emit all 9 kind entries (including zero
    // rows) in class order/kind order, then stable-sort by overhead desc.
    type BreakdownRow = (u8, String, String, u8, u64, i64);
    let mut per_source: [Vec<BreakdownRow>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..stats.per_class.len() {
        if canonical_of[i] != i {
            continue;
        }
        let pc = &stats.per_class[i];
        let (cat_id, _) = category_of(&pc.class_name);
        if cat_id == 0 {
            continue;
        }
        let class_name = java_class_name(&pc.class_name);
        let display = java_display_name(&pc.class_name);
        for k in 0..COLLECTION_KIND_COUNT {
            per_source[(cat_id - 1) as usize].push((
                cat_id,
                class_name.clone(),
                display.clone(),
                k as u8,
                aggs[i].kind_count[k],
                aggs[i].kind_size[k],
            ));
        }

        let total_objects = match cat_id {
            1 => aggs[i].instance_count,
            _ => aggs[i].array_count,
        };
        match cat_id {
            1 => {
                summary.collection_total_objects += total_objects;
                summary.collection_problem_count += aggs[i].bad_count;
                summary.collection_overhead += aggs[i].bad_size;
            }
            2 => {
                summary.obj_array_total_objects += total_objects;
                summary.obj_array_problem_count += aggs[i].bad_count;
                summary.obj_array_overhead += aggs[i].bad_size;
            }
            3 => {
                summary.prim_array_total_objects += total_objects;
                summary.prim_array_problem_count += aggs[i].bad_count;
                summary.prim_array_overhead += aggs[i].bad_size;
            }
            _ => {}
        }
    }

    {
        let mut out = BufWriter::with_capacity(1 << 18, File::create(dir.join("boxed_summary.tsv"))?);
        writeln!(
            out,
            "source\trow_kind\tentry_index\ttotal_object_count\ttotal_problem_count\ttotal_overhead\toverhead_bytes\tproblem_kind\tproblem_kind_text\tobject_count\tclass_name\tclass_display_name"
        )?;
        let sources: [(&str, u64, u64, i64); 3] = [
            ("collection", summary.collection_total_objects, summary.collection_problem_count, summary.collection_overhead),
            ("object_array", summary.obj_array_total_objects, summary.obj_array_problem_count, summary.obj_array_overhead),
            ("prim_array", summary.prim_array_total_objects, summary.prim_array_problem_count, summary.prim_array_overhead),
        ];
        for (idx, (source, objects, problems, overhead)) in sources.iter().enumerate() {
            writeln!(
                out,
                "{}\tsummary\t-1\t{}\t{}\t{}\t-\t-\t-\t-\t-\t-",
                source, objects, problems, overhead
            )?;
            // Stable sort by overhead desc, matching Java's
            // Collections.sort(... Long.signum(b - a)).
            per_source[idx].sort_by(|a, b| b.5.cmp(&a.5));
            for (entry_index, (_, class_name, display, kind, count, overhead_bytes)) in
                per_source[idx].iter().enumerate()
            {
                writeln!(
                    out,
                    "{}\tentry\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    source,
                    entry_index,
                    objects,
                    problems,
                    overhead,
                    overhead_bytes,
                    collection_problem_enum_name(*kind),
                    java_kind_name(*kind),
                    count,
                    class_name,
                    display
                )?;
            }
        }
    }

    Ok(summary)
}
