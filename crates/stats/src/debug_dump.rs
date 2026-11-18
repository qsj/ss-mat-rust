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

/// Disk-backed complete S14 membership. Records retain producer order and
/// duplicate multiplicity; callers may visit repeatedly without materializing it.
pub struct DupPrimMemberStore {
    file: File,
    count: u64,
    offset: u64,
}

impl DupPrimMemberStore {
    pub fn create(
        members: impl IntoIterator<Item = std::io::Result<DupPrimMemberDebug>>,
    ) -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let base = std::env::var_os("SS_MAT_TEMP_DIR")
            .filter(|v| !v.is_empty()).map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let (path, file) = loop {
            let serial = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!("ss-mat-members-{}-{}", std::process::id(), serial));
            match std::fs::OpenOptions::new().read(true).write(true).create_new(true).open(&path) {
                Ok(file) => break (path, file),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        };
        // Only this invocation's create_new file is unlinked. The descriptor
        // owns its lifetime, including early iterator/write failures.
        std::fs::remove_file(path)?;
        let mut count = 0;
        {
            let mut writer = BufWriter::with_capacity(64 << 10, &file);
            for member in members {
                let member = member?;
                writer.write_all(&member.obj_id.to_le_bytes())?;
                writer.write_all(&[u8::from(member.string_backing), u8::from(member.container_backing)])?;
                count += 1;
            }
            writer.flush()?;
        }
        Ok(Self { file, count, offset: 0 })
    }

    pub fn len(&self) -> u64 { self.count }
    pub fn is_empty(&self) -> bool { self.count == 0 }

    pub fn visit(
        &self,
        mut f: impl FnMut(DupPrimMemberDebug) -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let mut buffer = [0u8; 10 * 4096];
        let mut at = 0;
        while at < self.count {
            let records = (self.count - at).min(4096) as usize;
            let bytes = &mut buffer[..records * 10];
            self.file.read_exact_at(bytes, self.offset + at * 10)?;
            for raw in bytes.chunks_exact(10) {
                f(DupPrimMemberDebug {
                    obj_id: u64::from_le_bytes(raw[..8].try_into().unwrap()),
                    string_backing: raw[8] != 0,
                    container_backing: raw[9] != 0,
                })?;
            }
            at += records as u64;
        }
        Ok(())
    }
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

/// Threshold/candidate ground truth for Java's Section 15 scanner.
#[derive(Clone, Copy, Default)]
pub struct DupObjMetrics {
    pub total_bytes: i64,
    pub instance_total_bytes: i64,
    pub object_array_total_size: i64,
    pub primitive_array_total_size: i64,
    pub dup_threshold: i64,
    pub class_threshold: i64,
    pub candidate_count: usize,
}

/// One Java `DuplicateGroupEntry` of the S15 report.
#[derive(Clone)]
pub struct DupObjGroupDebug {
    pub class_name: String,
    pub group_id: u32,
    pub rep_obj_id: u64,
    pub count: usize,
    pub wasted_bytes: i64,
    pub body_size: i64,
    pub array_length: i64,
}

static DUP_OBJ_METRICS: OnceLock<Mutex<DupObjMetrics>> = OnceLock::new();
static DUP_OBJ_CANDIDATES: OnceLock<Mutex<Vec<(String, String, i64, i64, i64)>>> = OnceLock::new();
static DUP_OBJ_ARRAY_CLASSES: OnceLock<Mutex<Vec<(String, String, i64, i64, i64, i64)>>> = OnceLock::new();
static DUP_OBJ_GROUPS: OnceLock<Mutex<Vec<DupObjGroupDebug>>> = OnceLock::new();

fn obj_metrics_store() -> &'static Mutex<DupObjMetrics> {
    DUP_OBJ_METRICS.get_or_init(|| Mutex::new(DupObjMetrics::default()))
}

pub fn record_dup_obj_metrics(metrics: DupObjMetrics) {
    *obj_metrics_store().lock().unwrap() = metrics;
}

pub fn record_dup_obj_candidates(candidates: Vec<(String, String, i64, i64, i64)>) {
    *DUP_OBJ_CANDIDATES.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap() = candidates;
}

pub fn record_dup_obj_array_classes(rows: Vec<(String, String, i64, i64, i64, i64)>) {
    *DUP_OBJ_ARRAY_CLASSES.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap() = rows;
}

pub fn record_dup_obj_groups(groups: Vec<DupObjGroupDebug>) {
    *DUP_OBJ_GROUPS.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap() = groups;
}

struct StoredPrimGroup {
    metadata: DupPrimGroupDebug,
    members: DupPrimMemberStore,
}

impl std::ops::Deref for StoredPrimGroup {
    type Target = DupPrimGroupDebug;
    fn deref(&self) -> &Self::Target { &self.metadata }
}

struct PrimRegistry {
    rows: crate::problem_types::RecordStore<PrimDebugRow>,
    members: DupPrimMemberStore,
    classes: Vec<String>,
}
struct PrimDebugRow([u64; 9]);
impl crate::problem_types::StoredRecord for PrimDebugRow {
    const WIDTH: usize = 72;
    fn encode(&self,r:&mut [u8]) { for (raw,n) in r.chunks_exact_mut(8).zip(self.0) {raw.copy_from_slice(&n.to_le_bytes());} }
    fn decode(r:&[u8])->Self {let mut a=[0;9]; for (n,raw) in a.iter_mut().zip(r.chunks_exact(8)){*n=u64::from_le_bytes(raw.try_into().unwrap());} Self(a)}
}
impl PrimRegistry {
    fn new()->std::io::Result<Self> { Ok(Self {rows:crate::problem_types::RecordStore::new()?, members:DupPrimMemberStore::create(std::iter::empty())?,classes:Vec::new()}) }
    fn push(&mut self, metadata:DupPrimGroupDebug, members:DupPrimMemberStore)->std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        let class = if let Some(i)=self.classes.iter().position(|n| *n==metadata.class_name){i} else {self.classes.push(metadata.class_name);self.classes.len()-1};
        let offset=self.members.count*10;
        members.visit(|m| {
            let mut raw=[0;10];raw[..8].copy_from_slice(&m.obj_id.to_le_bytes());raw[8]=u8::from(m.string_backing);raw[9]=u8::from(m.container_backing);
            self.members.file.write_all_at(&raw,self.members.count*10)?; self.members.count+=1;Ok(())
        })?;
        self.rows.push(PrimDebugRow([class as u64,metadata.array_length as u64,metadata.element_size as u64,
            metadata.body_size as u64,metadata.count as u64,metadata.wasted_bytes as u64,metadata.content_hash,offset,members.count]))
    }
    fn get(&self,index:u64)->std::io::Result<StoredPrimGroup>{
        let r=self.rows.get(index)?.0;
        Ok(StoredPrimGroup {metadata:DupPrimGroupDebug {class_name:self.classes[r[0] as usize].clone(),array_length:r[1] as i32,
            element_size:r[2] as i32,body_size:r[3] as i64,count:r[4] as usize,wasted_bytes:r[5] as i64,content_hash:r[6],members:Vec::new()},
            members:DupPrimMemberStore {file:self.members.file.try_clone()?,count:r[8],offset:r[7]}})
    }
}

/// Add one group whose complete membership has already been spilled.
/// Group arrival order remains the stable tie-breaker in forensic output.
pub fn record_dup_prim_group_store(
    metadata: DupPrimGroupDebug,
    members: DupPrimMemberStore,
) -> std::io::Result<()> {
    if !metadata.members.is_empty() || metadata.count as u64 != members.len() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "inconsistent S14 streamed membership"));
    }
    let mut previous = None;
    members.visit(|member| {
        if previous.is_some_and(|addr| addr > member.obj_id) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "S14 member store is not address ordered"));
        }
        previous = Some(member.obj_id);
        Ok(())
    })?;
    let mut registry=group_store().lock().unwrap();
    if registry.is_none(){*registry=Some(PrimRegistry::new()?);}
    registry.as_mut().unwrap().push(metadata,members)?;
    Ok(())
}

static DUP_PRIM_GROUPS: OnceLock<Mutex<Option<PrimRegistry>>> = OnceLock::new();
static DUP_PRIM_TOTALS: OnceLock<Mutex<DupPrimScanTotals>> = OnceLock::new();

fn group_store() -> &'static Mutex<Option<PrimRegistry>> {
    DUP_PRIM_GROUPS.get_or_init(|| Mutex::new(None))
}

fn totals_store() -> &'static Mutex<DupPrimScanTotals> {
    DUP_PRIM_TOTALS.get_or_init(|| Mutex::new(DupPrimScanTotals::default()))
}

/// Store duplicate-group forensics. No-op unless the debug switch is on.
pub fn record_dup_prim_groups(groups: Vec<DupPrimGroupDebug>) {
    for mut metadata in groups {
        let mut members = std::mem::take(&mut metadata.members);
        members.sort_by_key(|m| m.obj_id);
        let members = DupPrimMemberStore::create(members.into_iter().map(Ok))
            .expect("spill S14 forensic members");
        record_dup_prim_group_store(metadata, members).expect("record S14 forensic members");
    }
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

    let store = group_store().lock().unwrap();
    let scratch=DebugScratch::new()?;
    let input=scratch.0.join("s14-input"); let sorted=scratch.0.join("s14-sorted");
    let group_count=store.as_ref().map_or(0,|s|s.rows.len());
    let mut writer=BufWriter::new(File::create(&input)?);
    if let Some(store)=store.as_ref(){
        let mut class_order:Vec<_>=(0..store.classes.len()).collect();
        class_order.sort_by(|a,b|store.classes[*a].cmp(&store.classes[*b]));
        for (i,row) in store.rows.iter().enumerate(){
            let r=row?.0;
            let rank=class_order.iter().position(|v| *v==r[0] as usize).unwrap() as u64;
            for n in [!(r[5]^(1u64<<63)),rank,(r[1] as i32 as i64 as u64)^(1u64<<63),r[6],i as u64]{writer.write_all(&n.to_le_bytes())?;}
        }
    }
    writer.flush()?;
    crate::problem_types::sort_record_keys(&input,&sorted,40,&[0,8,16,24,32])?;
    let ordered=File::open(sorted)?;
    let group_at=|i:u64|->std::io::Result<StoredPrimGroup>{
        use std::os::unix::fs::FileExt;
        let mut raw=[0;8];ordered.read_exact_at(&mut raw,i*40+32)?;
        store.as_ref().unwrap().get(u64::from_le_bytes(raw))
    };

    let mut summary = S14DumpSummary {
        groups: group_count as usize,
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

    for i in 0..group_count {
        let g=group_at(i)?;
        summary.arrays_in_groups += g.count;
        summary.waste_bytes += g.wasted_bytes;
        let mut live = 0usize;
        g.members.visit(|m| {
            if live_prim_obj_ids.contains(&m.obj_id) {
                live += 1;
            }
            if m.string_backing {
                summary.string_backing_members += 1;
            }
            if m.container_backing {
                summary.container_backing_members += 1;
            }
            Ok(())
        })?;
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
        for i in 0..group_count {
            let g=group_at(i)?;
            let group_id = i + 1;
            let (mut live, mut string_backing, mut container_backing) = (0, 0, 0);
            let mut rep = None;
            g.members.visit(|m| {
                live += usize::from(live_prim_obj_ids.contains(&m.obj_id));
                string_backing += usize::from(m.string_backing);
                container_backing += usize::from(m.container_backing);
                rep = Some(rep.map_or(m.obj_id, |r: u64| r.min(m.obj_id)));
                Ok(())
            })?;
            let rep = rep.unwrap_or(0);
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
        out.flush()?;
    }

    {
        let mut out = BufWriter::with_capacity(1 << 20, File::create(&members_path)?);
        writeln!(
            out,
            "group_id\tobj_id_hex\tobj_id_dec\tlive\tstring_backing\tcontainer_backing\tclass_name\tarray_length\tbody_size\tcontent_hash"
        )?;
        for i in 0..group_count {
            let g=group_at(i)?;
            let group_id = i + 1;
            g.members.visit(|m| {
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
                Ok(())
            })?;
        }
        out.flush()?;
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
    let mut per_class: std::collections::HashMap<String, (usize, usize, i64)> =
        std::collections::HashMap::new();
    for i in 0..group_count {
        let g=group_at(i)?;
        let e = per_class.entry(g.class_name.clone()).or_insert((0, 0, 0));
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
    let mut out = HashSet::new();
    traversal.for_each_in_dump_order(|_dump_rank, addr_rank, entry| {
        if entry.kind == 1 && traversal.visited.get(addr_rank) {
            out.insert(entry.obj_id);
        }
    });
    out
}

// ---------------------------------------------------------------------------
// Reference chains (Section 3 / Section 19)
// ---------------------------------------------------------------------------

fn chain_step_text(step: &crate::ref_chain::ChainStep) -> String {
    use crate::ref_chain::ChainStep;
    match step {
        ChainStep::Field { class_name, field_name } => format!("{class_name}.{field_name}"),
        ChainStep::ArrayElement { class_name } => format!("{class_name}[elements]"),
        ChainStep::Root { description } => format!("root:{description}"),
    }
}

/// One rendered row per chain group, ordered so two runs compare byte for byte.
///
/// The report groups samples in a hash map, so the order of groups of equal size
/// is not stable between runs; the dump sorts on the group content instead, which
/// keeps the comparison meaningful for everything the group actually holds.
fn chain_rows(store: &crate::ref_chain::RefChainStore, section: &str) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    let mut classes: Vec<(&u32, &crate::ref_chain::ClassRefChains)> = store.chains.iter().collect();
    classes.sort_by_key(|(idx, _)| **idx);
    for (class_idx, chains) in classes {
        for (kind, groups) in [
            ("expensive", &chains.expensive_fields),
            ("full", &chains.full_chains),
        ] {
            for group in groups.iter() {
                let path: Vec<String> = group.path.iter().map(chain_step_text).collect();
                let ids: Vec<String> = group.sample_obj_ids.iter().map(|id| format!("{id:x}")).collect();
                rows.push(format!(
                    "{section}\tclass={class_idx}\tkind={kind}\tsize={}\tcount={}\tids={}\tpath={}\tsamples={}",
                    group.size_bytes,
                    group.count,
                    ids.join(","),
                    path.join("<"),
                    group.samples.join("|"),
                ));
            }
        }
    }
    rows
}

/// Write the reference-chain store exactly as the traversal handed it over.
pub fn write_ref_chains(dir: &Path, store: &crate::ref_chain::RefChainStore) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut rows = chain_rows(store, "class");
    let mut humongous = Vec::new();
    for (kind, groups) in [
        ("expensive", &store.humongous_chains.expensive_fields),
        ("full", &store.humongous_chains.full_chains),
    ] {
        for group in groups.iter() {
            let path: Vec<String> = group.path.iter().map(chain_step_text).collect();
            let ids: Vec<String> = group.sample_obj_ids.iter().map(|id| format!("{id:x}")).collect();
            humongous.push(format!(
                "humongous\tkind={kind}\tsize={}\tcount={}\tids={}\tpath={}\tsamples={}",
                group.size_bytes,
                group.count,
                ids.join(","),
                path.join("<"),
                group.samples.join("|"),
            ));
        }
    }
    rows.sort();
    humongous.sort();
    let mut out = BufWriter::with_capacity(1 << 16, File::create(dir.join("ref_chains.tsv"))?);
    writeln!(out, "section\tclass\tkind\tsize\tcount\tids\tpath\tsamples")?;
    for row in &rows {
        writeln!(out, "{row}")?;
    }
    for row in &humongous {
        writeln!(out, "{row}")?;
    }
    out.flush()
}

// ---------------------------------------------------------------------------
// Aggregate problem forensics (S8/S9/S10/S11/S12 totals per class)
// ---------------------------------------------------------------------------

struct DebugScratch(std::path::PathBuf);
impl DebugScratch {
    fn new() -> std::io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!("ss-mat-debug-sort-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
    }
}
impl Drop for DebugScratch { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

pub struct StringDebugStore {
    rows: crate::problem_types::RecordStore<StringDebugRow>,
    details: Vec<DupStringGroup>,
}
struct StringDebugRow { count: u64, waste: i64, backing: u64, detail: u64, hash: u64 }
impl crate::problem_types::StoredRecord for StringDebugRow {
    const WIDTH: usize = 40;
    fn encode(&self, raw: &mut [u8]) {
        for (chunk, value) in raw.chunks_exact_mut(8).zip([self.count,self.waste as u64,self.backing,self.detail,self.hash]) { chunk.copy_from_slice(&value.to_le_bytes()); }
    }
    fn decode(raw: &[u8]) -> Self {
        let n = |i| u64::from_le_bytes(raw[i..i+8].try_into().unwrap());
        Self { count:n(0), waste:n(8) as i64, backing:n(16), detail:n(24), hash:n(32) }
    }
}
impl StringDebugStore {
    pub fn new() -> std::io::Result<Self> { Ok(Self { rows: crate::problem_types::RecordStore::new()?, details: Vec::new() }) }
    pub fn push(&mut self, g: &DupStringGroup) -> std::io::Result<()> {
        let detail = if !g.content.is_empty() || !g.sample_obj_ids.is_empty() {
            if self.details.len() >= 200 { return Err(std::io::Error::other("S8 detail exceeds baseline top200")); }
            self.details.push(g.clone()); self.details.len() as u64
        } else { 0 };
        self.rows.push(StringDebugRow { count:g.count as u64,waste:g.wasted_bytes,backing:g.backing_array_count as u64,detail,hash:fnv1a64(g.content.as_bytes()) })
    }
    fn group(&self, index: u64) -> std::io::Result<DupStringGroup> {
        let row = self.rows.get(index)?;
        Ok(if row.detail != 0 { self.details[row.detail as usize-1].clone() } else {
            DupStringGroup { content:String::new(), count:row.count as usize,wasted_bytes:row.waste,
                backing_array_count:row.backing as usize,sample_obj_ids:Vec::new() }
        })
    }
}
static STREAM_STRING_GROUPS: OnceLock<Mutex<Option<StringDebugStore>>> = OnceLock::new();
pub fn record_dup_string_store(store: StringDebugStore) {
    *STREAM_STRING_GROUPS.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(store);
}

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

/// Mirror of Java's `ContainerRegistry` registration: a class gets
/// `PerClassProblemStats(..., 1)` only when the corresponding descriptor
/// constructor completes. Field lookups follow `getFieldIndexOrThrow`, which
/// walks the superclass chain; alternative names are tried in order.
fn java_container_class_set(heap: &ss_heap_model::heap_dump::HeapDump) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    let mut addr_to_idx: std::collections::HashMap<u64, u32> =
        std::collections::HashMap::with_capacity(heap.classes.len());
    for c in &heap.classes {
        if c.class_object_id != 0 {
            addr_to_idx.insert(c.class_object_id, c.index);
        }
    }
    let idx_of = |name: &str| -> Option<u32> { heap.class_by_name.get(name).copied() };
    let find_field = |idx: u32, names: &[&str]| -> bool {
        let mut cur = Some(idx);
        while let Some(i) = cur {
            let c = &heap.classes[i as usize];
            if c.instance_fields.iter().any(|fd| names.iter().any(|n| fd.name.as_ref() == *n)) {
                return true;
            }
            cur = if c.super_class_id == 0 { None } else { addr_to_idx.get(&c.super_class_id).copied() };
        }
        false
    };
    let has_all = |idx: u32, groups: &[&[&str]]| -> bool {
        groups.iter().all(|names| find_field(idx, names))
    };
    // Java resolveClassInfoPair: first alternative present in the dump wins.
    let resolve = |alts: &[&str]| -> Option<u32> { alts.iter().find_map(|n| idx_of(n)) };
    let add = |out: &mut HashSet<String>, idx: u32| {
        out.insert(heap.classes[idx as usize].name.to_string());
    };
    let try_register = |out: &mut HashSet<String>,
                        container: &str,
                        groups: &[&[&str]],
                        entry: Option<&[&str]>,
                        entry_fields: &[&[&str]]| {
        let Some(c) = idx_of(container) else { return };
        if !has_all(c, groups) {
            return;
        }
        if let Some(alts) = entry {
            let Some(e) = resolve(alts) else { return };
            if !has_all(e, entry_fields) {
                return;
            }
        }
        add(out, c);
    };

    // registerArrayBasedCollection / registerCompactArray.
    try_register(&mut out, "java/util/ArrayList", &[&["size"], &["elementData", "array"]], None, &[]);
    try_register(&mut out, "java/util/Vector", &[&["elementCount"], &["elementData"]], None, &[]);
    try_register(&mut out, "java/util/Stack", &[&["elementCount"], &["elementData"]], None, &[]);
    try_register(&mut out, "java/util/IdentityHashMap", &[&["size"], &["table", "elementData"]], None, &[]);
    try_register(&mut out, "scala/collection/mutable/ArrayBuffer",
        &[&["size0", "scala$collection$mutable$ResizableArray$$size1"], &["array"]], None, &[]);

    // registerHashTableMap family.
    try_register(&mut out, "java/util/HashMap", &[&["size"], &["table"]],
        Some(&["java/util/HashMap$Entry", "java/util/HashMap$Node", "java/util/HashMap$HashMapEntry"]),
        &[&["next"], &["key", "referent"], &["value"]]);
    try_register(&mut out, "java/util/LinkedHashMap", &[&["size"], &["table"], &["header", "head"]],
        Some(&["java/util/LinkedHashMap$Entry", "java/util/LinkedHashMap$LinkedHashMapEntry", "java/util/LinkedHashMap$LinkedEntry"]),
        &[&["next"], &["key"], &["value"]]);
    try_register(&mut out, "java/util/Hashtable", &[&["count", "size"], &["table"]],
        Some(&["java/util/Hashtable$Entry", "java/util/Hashtable$HashtableEntry"]),
        &[&["next"], &["key", "referent"], &["value"]]);
    try_register(&mut out, "java/lang/System$PropertiesWithNonOverrideableDefaults",
        &[&["count", "size"], &["table"]],
        Some(&["java/util/Hashtable$Entry", "java/util/Hashtable$HashtableEntry"]),
        &[&["next"], &["key", "referent"], &["value"]]);
    try_register(&mut out, "java/lang/ThreadLocal$ThreadLocalMap", &[&["size"], &["table"]],
        Some(&["java/lang/ThreadLocal$ThreadLocalMap$Entry"]),
        &[&["key", "referent"], &["value"]]);

    // java.util.Properties: Hashtable path when it has no `map` field,
    // otherwise the wrapped-map descriptor (which needs the `map` field).
    if let Some(p) = idx_of("java/util/Properties") {
        if find_field(p, &["map"]) {
            add(&mut out, p);
        } else {
            try_register(&mut out, "java/util/Properties", &[&["count", "size"], &["table"]],
                Some(&["java/util/Hashtable$Entry", "java/util/Hashtable$HashtableEntry"]),
                &[&["next"], &["key", "referent"], &["value"]]);
        }
    }

    // ConcurrentHashMap: CHMv8 when Node exists, JDK7 segments otherwise.
    if let Some(node) = idx_of("java/util/concurrent/ConcurrentHashMap$Node") {
        let chm_ok = idx_of("java/util/concurrent/ConcurrentHashMap").is_some_and(|c| {
            has_all(c, &[&["table"], &["nextTable"], &["baseCount"], &["counterCells"]])
        });
        let node_ok = has_all(node, &[&["next"], &["key"], &["val"]]);
        let mut extras_ok = true;
        if let Some(f) = idx_of("java/util/concurrent/ConcurrentHashMap$ForwardingNode") {
            extras_ok &= find_field(f, &["nextTable"]);
        }
        if let Some(t) = idx_of("java/util/concurrent/ConcurrentHashMap$TreeNode") {
            extras_ok &= has_all(t, &[&["left"], &["right"]]);
        }
        if let Some(t) = idx_of("java/util/concurrent/ConcurrentHashMap$TreeBin") {
            extras_ok &= has_all(t, &[&["root"], &["first"]]);
        }
        if let Some(cc) = idx_of("java/util/concurrent/ConcurrentHashMap$CounterCell") {
            extras_ok &= find_field(cc, &["value"]);
        }
        if chm_ok && node_ok && extras_ok {
            add(&mut out, idx_of("java/util/concurrent/ConcurrentHashMap").unwrap());
        }
    } else if let Some(chm) = idx_of("java/util/concurrent/ConcurrentHashMap") {
        let seg = idx_of("java/util/concurrent/ConcurrentHashMap$Segment");
        let entry = resolve(&[
            "java/util/concurrent/ConcurrentHashMap$HashEntry",
            "java/util/concurrent/ConcurrentHashMap$Node",
        ]);
        if let (Some(seg), Some(entry)) = (seg, entry) {
            if find_field(chm, &["segments"])
                && has_all(seg, &[&["table"], &["count"], &["sync"]])
                && has_all(entry, &[&["next"], &["key"], &["value"]])
            {
                add(&mut out, chm);
            }
        }
    }

    // Wrapped maps (HashSet/LinkedHashSet/TreeSet) and wrapped collections.
    for (container, field) in [
        ("java/util/HashSet", &["map", "backingMap"][..]),
        ("java/util/LinkedHashSet", &["map", "backingMap"][..]),
        ("java/util/TreeSet", &["m", "backingMap"][..]),
    ] {
        if let Some(c) = idx_of(container) {
            if find_field(c, field) {
                add(&mut out, c);
            }
        }
    }

    // Linked-list containers.
    let linked_ok = |container: &str,
                     size_fields: &[&str],
                     head_fields: &[&str],
                     node_alts: &[&str],
                     item_fields: &[&str]| -> bool {
        let (Some(c), Some(node)) = (idx_of(container), resolve(node_alts)) else { return false };
        find_field(c, size_fields)
            && find_field(c, head_fields)
            && find_field(node, item_fields)
            && find_field(node, &["next"])
    };
    if linked_ok("java/util/LinkedList", &["size"], &["first", "header", "voidLink"],
        &["java/util/LinkedList$Node", "java/util/LinkedList$Entry", "java/util/LinkedList$Link"],
        &["item", "element", "data"])
    {
        add(&mut out, idx_of("java/util/LinkedList").unwrap());
    }
    if idx_of("java/util/concurrent/atomic/AtomicInteger").is_some()
        && linked_ok("java/util/concurrent/LinkedBlockingQueue", &["count"], &["head"],
            &["java/util/concurrent/LinkedBlockingQueue$Node"], &["item"])
    {
        add(&mut out, idx_of("java/util/concurrent/LinkedBlockingQueue").unwrap());
    }
    if linked_ok("java/util/concurrent/LinkedBlockingDeque", &["count"], &["first"],
        &["java/util/concurrent/LinkedBlockingDeque$Node"], &["item"])
    {
        add(&mut out, idx_of("java/util/concurrent/LinkedBlockingDeque").unwrap());
    }

    // TreeMap and its submap wrappers.
    let mut tree_ok = false;
    if let (Some(c), Some(e)) = (
        idx_of("java/util/TreeMap"),
        resolve(&["java/util/TreeMap$Entry", "java/util/TreeMap$TreeMapEntry", "java/util/TreeMap$Node"]),
    ) {
        if has_all(c, &[&["size"], &["root"]])
            && has_all(e, &[&["key"], &["value"], &["left"], &["right"]])
        {
            tree_ok = true;
            add(&mut out, c);
        }
    }
    if tree_ok {
        for wrapper in ["java/util/TreeMap$AscendingSubMap", "java/util/TreeMap$DescendingSubMap"] {
            if let Some(c) = idx_of(wrapper) {
                if find_field(c, &["m"]) {
                    add(&mut out, c);
                }
            }
        }
    }

    // WeakHashMap (needs ReferenceQueue.head).
    if let (Some(c), Some(e), Some(q)) = (
        idx_of("java/util/WeakHashMap"),
        resolve(&["java/util/WeakHashMap$Entry"]),
        idx_of("java/lang/ref/ReferenceQueue"),
    ) {
        if has_all(c, &[&["size", "elementCount"], &["table", "elementData"], &["queue", "referenceQueue"]])
            && has_all(e, &[&["next"], &["referent"], &["value"]])
            && find_field(q, &["head"])
        {
            add(&mut out, c);
        }
    }

    // ConcurrentSkipListMap.
    if let (Some(c), Some(n), Some(i)) = (
        idx_of("java/util/concurrent/ConcurrentSkipListMap"),
        resolve(&["java/util/concurrent/ConcurrentSkipListMap$Node"]),
        idx_of("java/util/concurrent/ConcurrentSkipListMap$Index"),
    ) {
        if find_field(c, &["head"])
            && has_all(n, &[&["key"], &["value", "val"], &["next"]])
            && has_all(i, &[&["node"], &["down"], &["right"]])
        {
            add(&mut out, c);
        }
    }

    // Guava RegularImmutableMap (FlatEntryTableDescriptor).
    if let (Some(c), Some(e)) = (
        idx_of("com/google/common/collect/RegularImmutableMap"),
        resolve(&[
            "com/google/common/collect/RegularImmutableMap$NonTerminalEntry",
            "com/google/common/collect/RegularImmutableMap$TerminalEntry",
        ]),
    ) {
        if has_all(c, &[&["entries"], &["table"]]) && has_all(e, &[&["key"], &["value"]]) {
            add(&mut out, c);
        }
    }

    out
}

/// Java's `PerClassProblemStats.problemObjectCount`: 1=collection,
/// 2=object_array, 3=primitive_array, 0=other.
fn category_of(class_name: &str, containers: &HashSet<String>) -> (u8, &'static str) {
    if synthetic_to_hprof(class_name).is_some() || is_single_dim_prim_hprof(class_name) {
        (3, "primitive_array")
    } else if is_multi_dim_prim_hprof(class_name) {
        // Java leaves 2-D primitive arrays with plain PerClassStats.
        (0, "other")
    } else if class_name.starts_with('[') || class_name.ends_with("[]") {
        (2, "object_array")
    } else if containers.contains(class_name) {
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
pub fn java_class_display_name(raw: &str) -> String {
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
    heap: &ss_heap_model::heap_dump::HeapDump,
) -> std::io::Result<ProblemDumpSummary> {
    use std::sync::atomic::Ordering::Relaxed;
    use crate::problem_types::*;

    std::fs::create_dir_all(dir)?;

    // Java only assigns problemObjectCount=1 to classes whose
    // ContainerRegistry descriptor registered; derive that exact set.
    let containers = java_container_class_set(heap);

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
        let store=stats.bad_collection_store.lock().unwrap();
        for r in store.as_ref().into_iter().flat_map(|s|s.iter()).chain(records.iter().filter(|_|store.is_none()).cloned().map(Ok)) {
            let r=r?;
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
    let mut container_backing=HashSet::new();
    {
        let records=stats.bad_collection_records.lock().unwrap();
        let store=stats.bad_collection_store.lock().unwrap();
        for r in store.as_ref().into_iter().flat_map(|s|s.iter()).chain(records.iter().filter(|_|store.is_none()).cloned().map(Ok)) {
            let id=r?.backing_array_id;if id!=0 {container_backing.insert(id);}
        }
    }

    {
        let records = stats.bad_array_records.lock().unwrap();
            let store = stats.bad_array_store.lock().unwrap();
        for r in store.as_ref().into_iter().flat_map(|s| s.iter()).chain(records.iter().filter(|_| store.is_none()).cloned().map(Ok)) {
            let r = r?;
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
            let store = stats.bad_prim_array_store.lock().unwrap();
        for r in store.as_ref().into_iter().flat_map(|s| s.iter()).chain(records.iter().filter(|_| store.is_none()).cloned().map(Ok)) {
            let r = r?;
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
            let (cat_id, _) = category_of(&pc.class_name, &containers);
            // Java's PerClassStats.instanceCount includes arrays for array
            // classes (HprofDumpReader increments it in addObjectArray /
            // addPrimitiveArray); Rust keeps them in array_live_count.
            let instance_count = aggs[*i].instance_count + aggs[*i].array_count;
            write!(
                out,
                "{}\t{}\t{}\t{}\t{}",
                class_name,
                java_class_display_name(&pc.class_name),
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
        out.flush()?;
    }

    // ---- bad_collections.tsv (forensic S9 record dump) ----
    {
        let records = stats.bad_collection_records.lock().unwrap();
        let store=stats.bad_collection_store.lock().unwrap();
        let mut out = BufWriter::with_capacity(1 << 20, File::create(dir.join("bad_collections.tsv"))?);
        writeln!(
            out,
            "class_name\tobj_index\tobj_address_hex\tkind\tsize\tbody_size\twaste\toverhead\tis_collection\tcategory_count"
        )?;
        for r in store.as_ref().into_iter().flat_map(|s|s.iter()).chain(records.iter().filter(|_|store.is_none()).cloned().map(Ok)) {
            let r=r?;
            if r.kind == crate::problem_types::COLLECTION_KIND_NONE {
                continue;
            }
            let class_name = stats
                .per_class
                .get(r.class_idx as usize)
                .map(|pc| java_class_name(&pc.class_name))
                .unwrap_or_else(|| "unknown".to_string());
            let kind = crate::problem_types::collection_problem_enum_name(r.kind);
            writeln!(
                out,
                "{}\t-1\t{:x}\t{}\t{}\t{}\t{}\t-\t-\t-",
                class_name, r.obj_id, kind, r.size, r.body, r.waste
            )?;
        }
        out.flush()?;
    }

    // ---- s8_summary.tsv / s8_groups.tsv ----
    let report = stats.string_dup.get_report();
    let streamed = STREAM_STRING_GROUPS.get_or_init(|| Mutex::new(None)).lock().unwrap();
    let mut groups = if streamed.is_some() { Vec::new() } else { string_group_store().lock().unwrap().clone() };
    groups.sort_by(|a, b| {
        b.wasted_bytes
            .cmp(&a.wasted_bytes)
            .then_with(|| fnv1a64(a.content.as_bytes()).cmp(&fnv1a64(b.content.as_bytes())))
            .then_with(|| b.count.cmp(&a.count))
    });

    let scratch = DebugScratch::new()?;
    let index_input = scratch.0.join("s8-index-input");
    let index_sorted = scratch.0.join("s8-index-sorted");
    let mut sorted_indices = None;
    if let Some(store) = streamed.as_ref() {
        let mut writer = BufWriter::new(File::create(&index_input)?);
        for (index, row) in store.rows.iter().enumerate() {
            let row = row?;
            for value in [!(row.waste as u64 ^ (1u64<<63)), row.hash, !row.count, index as u64] {
                writer.write_all(&value.to_le_bytes())?;
            }
        }
        writer.flush()?;
        crate::problem_types::sort_record_keys(&index_input, &index_sorted, 32, &[0,8,16,24])?;
        sorted_indices = Some(std::io::BufReader::new(File::open(&index_sorted)?));
    }

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
        out.flush()?;
    }

    {
        let mut out = BufWriter::with_capacity(1 << 20, File::create(dir.join("s8_groups.tsv"))?);
        writeln!(
            out,
            "group_index\trepresentative_index\tobject_size\tunique_value_count\tduplicate_count\ttotal_waste\tgroup_id\trepresentative_object_group_id\trepresentative_class\trepresentative_body_size\trep_obj_id_hex\tcontent_hash\tcontent_len_chars\tcontent_preview\tsample_obj_ids_hex"
        )?;
        for i in 0..streamed.as_ref().map_or(groups.len(), |s| s.rows.len() as usize) {
            let owned;
            let g = if let Some(store) = streamed.as_ref() {
                use std::io::Read;
                let mut raw = [0u8;32]; sorted_indices.as_mut().unwrap().read_exact(&mut raw)?;
                owned = store.group(u64::from_le_bytes(raw[24..32].try_into().unwrap()))?;
                &owned
            } else { &groups[i] };
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
        out.flush()?;
    }

    // ---- s15_dup_obj_arrays.tsv (schema mirrors Java DebugDumper) ----
    {
        let report = stats.dup_obj_arrays.get_report();
        let metrics = *obj_metrics_store().lock().unwrap();
        let mut out = BufWriter::with_capacity(1 << 16, File::create(dir.join("s15_dup_obj_arrays.tsv"))?);
        writeln!(
            out,
            "row_kind\tentry_index\tclass_name\tclass_display_name\tinstance_count\tduplicate_count\twasted_bytes\trepresentative_index\trepresentative_address_hex\tgroup_id\trepresentative_class\trepresentative_body_size\trepresentative_length\treport_present\ttotal_duplicate_count\ttotal_group_count\ttotal_waste_bytes"
        )?;
        for (name, value) in [
            ("total_bytes", metrics.total_bytes),
            ("instance_total_bytes", metrics.instance_total_bytes),
            ("object_array_total_size", metrics.object_array_total_size),
            ("primitive_array_total_size", metrics.primitive_array_total_size),
            ("dup_threshold", metrics.dup_threshold),
            ("class_percent_threshold", metrics.class_threshold),
            ("candidate_class_count", metrics.candidate_count as i64),
        ] {
            writeln!(out, "metric\t-1\t{}\t-\t-\t{}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-", name, value)?;
        }
        let candidates = DUP_OBJ_CANDIDATES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clone();
        for (name, display, instances, overhead, self_size) in candidates {
            writeln!(
                out,
                "candidate\t-1\t{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-",
                name, display, instances, overhead, self_size
            )?;
        }
        let near = DUP_OBJ_ARRAY_CLASSES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clone();
        for (name, display, instances, overhead, self_size, aligned) in near {
            writeln!(
                out,
                "obj_array_class\t-1\t{}\t{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t-\t-\t-\t-",
                name, display, instances, overhead, self_size, aligned
            )?;
        }
        writeln!(
            out,
            "summary\t-1\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t{}\t{}\t{}\t{}",
            if report.report_present { 1 } else { 0 },
            report.total_count,
            report.unique_count,
            report.wasted_bytes
        )?;
        if report.report_present {
            for (i, entry) in report.per_class.iter().enumerate() {
                writeln!(
                    out,
                    "class\t{}\t{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-",
                    i,
                    entry.class_name.replace('/', "."),
                    java_class_display_name(&entry.class_name),
                    entry.instance_count,
                    entry.unique_count,
                    entry.wasted_bytes
                )?;
            }
            let mut groups = DUP_OBJ_GROUPS
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap()
                .clone();
            groups.sort_by(|a, b| {
                b.wasted_bytes
                    .cmp(&a.wasted_bytes)
                    .then_with(|| a.class_name.cmp(&b.class_name))
                    .then_with(|| a.rep_obj_id.cmp(&b.rep_obj_id))
            });
            for (i, group) in groups.iter().enumerate() {
                writeln!(
                    out,
                    "group\t{}\t-\t-\t-\t{}\t{}\t-\t{:x}\t0\t{}\t{}\t{}\t-\t-\t-\t-",
                    i,
                    group.count,
                    group.wasted_bytes,
                    group.rep_obj_id,
                    java_class_display_name(&group.class_name),
                    group.body_size,
                    group.array_length
                )?;
            }
        }
        out.flush()?;
    }

    // ---- boxed_summary.tsv (schema mirrors Java DebugDumper) ----
    let mut summary = ProblemDumpSummary {
        s8_total_strings: report.total_strings,
        s8_unique_strings: report.unique_strings,
        s8_duplicate_values: report.duplicate_group_count,
        s8_wasted_bytes: report.wasted_bytes,
        s8_groups_dumped: streamed.as_ref().map_or(groups.len(), |s| s.rows.len() as usize),
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
        let (cat_id, _) = category_of(&pc.class_name, &containers);
        if cat_id == 0 {
            continue;
        }
        let class_name = java_class_name(&pc.class_name);
        let display = java_class_display_name(&pc.class_name);
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
        out.flush()?;
    }

    Ok(summary)
}

#[cfg(test)]
mod member_store_tests {
    #[test]
    fn truncated_debug_members_fail_without_fabricating_rows() {
        let store=super::DupPrimMemberStore::create([Ok(super::DupPrimMemberDebug {obj_id:4,string_backing:true,container_backing:false})]).unwrap();
        store.file.set_len(1).unwrap();
        assert_eq!(store.visit(|_|panic!("partial member emitted")).unwrap_err().kind(),std::io::ErrorKind::UnexpectedEof);
    }

    use super::*;

    #[test]
    fn preserves_order_multiplicity_flags_and_repeated_visits() {
        let store = DupPrimMemberStore::create((0..20000u64).map(|n| Ok(DupPrimMemberDebug {
            obj_id: (20000 - n) / 3,
            string_backing: n % 2 == 0,
            container_backing: n % 5 == 0,
        }))).unwrap();
        assert_eq!(store.len(), 20000);
        for _ in 0..2 {
            let mut n = 0;
            store.visit(|m| {
                assert_eq!(m.obj_id, (20000 - n) / 3);
                assert_eq!(m.string_backing, n % 2 == 0);
                assert_eq!(m.container_backing, n % 5 == 0);
                n += 1;
                Ok(())
            }).unwrap();
            assert_eq!(n, 20000);
        }
    }

    #[test]
    fn handles_empty_and_propagates_producer_and_consumer_errors() {
        let empty = DupPrimMemberStore::create(std::iter::empty()).unwrap();
        assert!(empty.is_empty());
        empty.visit(|_| panic!("empty store emitted a member")).unwrap();
        assert!(DupPrimMemberStore::create([Err(std::io::Error::other("producer"))]).is_err());
        let one = DupPrimMemberStore::create([Ok(DupPrimMemberDebug {
            obj_id: 0, string_backing: false, container_backing: false,
        })]).unwrap();
        assert_eq!(one.visit(|_| Err(std::io::Error::other("consumer"))).unwrap_err().to_string(), "consumer");
    }
}

#[cfg(test)]
mod bundle_flush_tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn empty_heap() -> ss_heap_model::heap_dump::HeapDump {
        ss_heap_model::heap_dump::HeapDump {
            classes:Vec::new(),class_by_name:Default::default(),roots:Vec::new(),total_object_count:0,
            string_table:Default::default(),identifier_size:8,oop_size:4,alignment:8,header_size:12,
            per_class_inst_count:Vec::new(),per_class_inst_bytes:Vec::new(),stack_traces:Default::default(),
            thread_serial_map:Default::default(),thread_info:Vec::new(),timestamp_millis:0,obj_dir:None,
        }
    }

    #[test]
    fn empty_s14_bundle_reports_buffered_write_failures() {
        for name in ["s14_groups.tsv", "s14_members.tsv"] {
            let scratch=DebugScratch::new().unwrap();
            symlink("/dev/full",scratch.0.join(name)).unwrap();
            let error=write_s14_bundle(&scratch.0,&HashSet::new()).err().expect("flush must fail");
            assert_eq!(error.raw_os_error(),Some(28));
        }
    }

    #[test]
    fn empty_problem_bundle_reports_every_buffered_write_failure() {
        let heap=empty_heap();let stats=StatsEngine::new(&[]);
        for name in ["problem_stats.tsv","bad_collections.tsv","s8_summary.tsv","s8_groups.tsv","s15_dup_obj_arrays.tsv","boxed_summary.tsv"] {
            let scratch=DebugScratch::new().unwrap();
            symlink("/dev/full",scratch.0.join(name)).unwrap();
            let error=write_problem_bundle(&scratch.0,&stats,&heap).err().expect("flush must fail");
            assert_eq!(error.raw_os_error(),Some(28),"{name}");
        }
    }
}
