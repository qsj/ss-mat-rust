//! Forensic TSV writers for GC roots and the BFS object table.
//!
//! Enabled only when `SS_MAT_DEBUG_DIR` is set; otherwise none of this runs.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use ss_heap_model::heap_dump::HeapDump;
use ss_heap_model::root::RootType;

use crate::heap_traverser::{compute_obj_shallow, array_shallow_size};
use ss_stats::traversal_data::ObjectEntry;

/// One row of `roots.tsv`, captured while the BFS seeds its queue.
pub struct RootRow {
    pub root_index: usize,
    pub root_type: RootType,
    pub java_category: &'static str,
    pub object_id: u64,
    pub thread_serial: Option<u32>,
    pub frame_index: Option<u32>,
    /// Resolved BFS object index, or -1 when the address is not in the object table.
    pub obj_idx: i64,
    pub class_name: String,
    /// -2 unresolved, -1 normal, 0 dropped, 1 deferred.
    pub priority: i32,
    pub action: &'static str,
}

/// Map Rust's HPROF root tags to the names Java's `RootType` uses, so root
/// distributions can be compared directly.
pub fn java_root_category(rt: RootType) -> &'static str {
    match rt {
        RootType::JniGlobal => "JNI_GLOBAL",
        RootType::JniLocal => "JNI_LOCAL",
        RootType::JavaStackFrame => "JAVA_LOCAL",
        RootType::NativeStack => "NATIVE_STACK",
        RootType::SystemClass => "SYSTEM_CLASS",
        RootType::ThreadBlock => "THREAD_BLOCK",
        RootType::BusyMonitor => "BUSY_MONITOR",
        // Java's HPROF reader maps tag 8 (THREAD_OBJECT) to THREAD_BLOCK.
        RootType::ThreadObj => "THREAD_BLOCK",
        RootType::Unknown => "UNKNOWN",
    }
}

fn buf(path: &Path) -> std::io::Result<BufWriter<File>> {
    Ok(BufWriter::with_capacity(1 << 20, File::create(path)?))
}

pub fn write_roots(dir: &Path, rows: &[RootRow]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut out = buf(&dir.join("roots.tsv"))?;
    writeln!(
        out,
        "root_index\troot_type\tjava_category\tobject_id_hex\tobject_id_dec\tthread_serial\tframe_index\tresolved_obj_idx\tclass_name\tpriority\taction"
    )?;
    for r in rows {
        writeln!(
            out,
            "{}\t{:?}\t{}\t0x{:x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.root_index,
            r.root_type,
            r.java_category,
            r.object_id,
            r.object_id,
            r.thread_serial.map(|v| v.to_string()).unwrap_or_default(),
            r.frame_index.map(|v| v.to_string()).unwrap_or_default(),
            r.obj_idx,
            r.class_name,
            r.priority,
            r.action,
        )?;
    }
    Ok(())
}

pub fn write_roots_summary(dir: &Path, rows: &[RootRow]) -> std::io::Result<()> {
    use std::collections::BTreeMap;

    // java_category -> [total, unresolved, dropped, already_visited, normal, deferred]
    let mut by_cat: BTreeMap<&'static str, [usize; 6]> = BTreeMap::new();
    let mut java_static = 0usize;
    for r in rows {
        let e = by_cat.entry(r.java_category).or_insert([0; 6]);
        e[0] += 1;
        match r.action {
            "unresolved" => e[1] += 1,
            "dropped_priority0" => e[2] += 1,
            "already_visited" => e[3] += 1,
            "seeded_normal" => e[4] += 1,
            "seeded_deferred" => e[5] += 1,
            _ => {}
        }
        if r.java_category == "JAVA_STATIC" {
            java_static += 1;
        }
    }

    let mut text = String::new();
    text.push_str("SS-Mat Rust GC root forensics\n");
    text.push_str(
        "action: unresolved = address not in BFS object table; dropped_priority0 = classifyRootPriority==0; \
         already_visited = target already seeded by an earlier root; seeded_normal/deferred = entered BFS queue\n",
    );
    text.push_str(
        "note: jvm-hprof 0.1 does not expose JAVA_STATIC roots (HPROF tag 0x20); static reachability is \
         simulated with post-BFS display-only roots, so JAVA_STATIC is expected to be absent here\n",
    );
    text.push_str(&format!("java_static_rows\t{}\n\n", java_static));
    text.push_str("java_category\ttotal\tunresolved\tdropped_priority0\talready_visited\tseeded_normal\tseeded_deferred\n");
    for (cat, e) in &by_cat {
        text.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            cat, e[0], e[1], e[2], e[3], e[4], e[5]
        ));
    }
    let totals = |idx: usize| by_cat.values().map(|e| e[idx]).sum::<usize>();
    text.push_str(&format!(
        "TOTAL\t{}\t{}\t{}\t{}\t{}\t{}\n",
        rows.len(),
        totals(1),
        totals(2),
        totals(3),
        totals(4),
        totals(5)
    ));
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("roots_summary.txt"), text)?;
    Ok(())
}

fn kind_label(kind: u8) -> &'static str {
    match kind {
        0 => "regular",
        1 => "primitive_array",
        2 => "object_array",
        3 => "class_dump",
        _ => "unknown",
    }
}

fn class_name(class_idx: u32, heap: &HeapDump) -> String {
    if (class_idx as usize) < heap.classes.len() {
        heap.classes[class_idx as usize].name.replace('/', ".")
    } else {
        "unknown".to_string()
    }
}

/// Write one row per BFS object table entry (2.7M rows on the reference dump).
pub fn write_objects(
    dir: &Path,
    objects: &[ObjectEntry],
    visited: &[bool],
    heap: &HeapDump,
    class_aligned_body: &[i64],
    live_count: u64,
    garbage_count: u64,
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut out = buf(&dir.join("objects.tsv"))?;
    writeln!(
        out,
        "obj_idx\tobject_id_hex\tobject_id_dec\tkind\tkind_label\tclass_idx\tclass_name\tarray_length\telement_size\tbody_size\tshallow_size\tlive\tcounted_live"
    )?;

    let oop = heap.oop_size as i32;
    let header = heap.header_size as i64;
    for (idx, obj) in objects.iter().enumerate() {
        let live = visited.get(idx).copied().unwrap_or(false);
        let shallow: i64 = match obj.kind {
            0 => class_aligned_body
                .get(obj.class_idx as usize)
                .copied()
                .unwrap_or_else(|| ((header + obj.body_size as i64 + 7) & !7)),
            1 => compute_obj_shallow(obj, oop, header),
            2 => array_shallow_size(obj.array_length, oop, header),
            // ClassDump: keep the same value the BFS live loop accumulates
            // (array_shallow_size with length 0), which is not a Java object size.
            _ => array_shallow_size(0, oop, header),
        };
        // Java's scannedObjectCount excludes ClassDump records.
        let counted_live = live && obj.kind != 3;
        writeln!(
            out,
            "{}\t0x{:x}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            idx,
            obj.obj_id,
            obj.obj_id,
            obj.kind,
            kind_label(obj.kind),
            obj.class_idx,
            class_name(obj.class_idx, heap),
            obj.array_length,
            obj.element_size,
            obj.body_size,
            shallow,
            if live { 1 } else { 0 },
            if counted_live { 1 } else { 0 },
        )?;
    }
    out.flush()?;

    // A tiny separate file keeps the counts next to the table for quick diffs.
    let summary = format!(
        "objects_total\t{}\nlive_counted\t{}\ngarbage_counted\t{}\nclass_dump_rows\t{}\n",
        objects.len(),
        live_count,
        garbage_count,
        objects.iter().filter(|o| o.kind == 3).count()
    );
    std::fs::write(dir.join("objects_summary.txt"), summary)?;
    Ok(())
}
