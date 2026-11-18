#![allow(dead_code, unused_variables, unused_assignments, unused_imports, unused_mut, unused_parens)]
mod flags;
mod progress;

use anyhow::{Context, Result};

#[global_allocator]
static GLOBAL_ALLOC: ss_support::mem_trace::CountingAlloc = ss_support::mem_trace::CountingAlloc;
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn main() -> Result<()> {
    // Install a panic hook that turns panics (e.g. from the jvm-hprof parser
    // on malformed input) into a clean error message instead of a backtrace.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".to_string()
        };
        let loc = info.location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        eprintln!(
            "\nError: analysis aborted unexpectedly.\n  Reason: {}\n  At: {}\n  This usually means the HPROF file is corrupted or uses an unsupported format.\n  Run with RUST_BACKTRACE=1 for a full backtrace.",
            msg, loc
        );
        // Do NOT call default_hook — we've replaced the scary output.
        let _ = default_hook;
    }));

    eprintln!("SS-Mat version 0.1.0 (Rust)");

    let cli = flags::Cli::parse();

    if cli.heap_dump.is_none() {
        flags::print_help();
        return Ok(());
    }

    let heap_dump_path = cli.heap_dump.as_ref().context(
        "Heap dump file path is required.\nUsage: ss-mat <heap_dump.hprof> [report.html] [options]",
    )?;

    validate_inputs(heap_dump_path)?;

    let report_path = cli
        .report_output
        .clone()
        .unwrap_or_else(|| derive_report_path(heap_dump_path));

    let flags = flags::Flags::from_cli(&cli);

    eprintln!("Heap dump: {}", heap_dump_path.display());
    eprintln!("Report output: {}", report_path.display());

    let run_start = Instant::now();
    ss_support::mem_trace::trace("start");

    // Phase 1: Parse HPROF file
    eprintln!("\n=== Phase 1: Parsing HPROF ===");
    let t = Instant::now();
    let temp_root = flags
        .temp_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("ss-mat-{}", std::process::id())));
    let workspace = ss_hprof_parser::objdir_build::TempWorkspace::create(&temp_root)
        .with_context(|| format!("cannot prepare temp dir {}", temp_root.display()))?;
    // Every writer in the pipeline takes its scratch directory from this variable.
    // Pointing it at the workspace is what gives the run's temp files one owner:
    // the BFS spilling used to fall back to its own /tmp/ss-mat-<pid> and outlive
    // the run whenever the workspace was somewhere else.
    std::env::set_var("SS_MAT_TEMP_DIR", workspace.path());
    let _temp_files = TempFiles::new(workspace.path().to_path_buf());
    let obj_cfg = ss_hprof_parser::objdir_build::ObjDirConfig {
        temp_dir: workspace.path().to_path_buf(),
        cache_bytes: (flags.cache_mb.max(1) as usize) << 20,
        disk_budget_bytes: flags.disk_budget_gb.saturating_mul(1 << 30),
    };

    let heap = ss_hprof_parser::hprof_wrapper::parse_hprof_with_config(heap_dump_path, &obj_cfg)
        .map_err(|e| friendly_hprof_error(&e, heap_dump_path))?;
    eprintln!(
        "  Classes: {}, Objects: {}, Roots: {}, Strings: {}",
        heap.classes.len(),
        heap.total_object_count,
        heap.roots.len(),
        heap.string_table.len()
    );
    print_elapsed("Phase 1", t.elapsed());
    ss_support::mem_trace::trace("phase1-parsed");

    // Phase 2: Initialize StatsEngine with per-class data
    eprintln!("\n=== Phase 2: Statistics Engine ===");
    let t = Instant::now();
    let stats = ss_stats::engine::StatsEngine::new(&heap.classes);
    populate_per_class_stats(&heap, &stats);
    print_elapsed("Phase 2", t.elapsed());

    // Phase 3: Linear scan
    eprintln!("\n=== Phase 3: Linear Scan ===");
    let t = Instant::now();
    let heap_arc = Arc::new(heap);
    let stats_arc = Arc::new(stats);

    ss_traversal::linear_scan::configure_thread_pool(flags.max_scan_threads)?;

    let coordinator = ss_traversal::linear_scan::LinearScanCoordinator::with_config(
        Arc::clone(&heap_arc),
        Arc::clone(&stats_arc),
        ss_traversal::linear_scan::ScanConfig {
            min_trailing_zeroes_percent: flags.min_trailing_zeroes_percent,
            min_sparse_zeroes_percent: flags.min_sparse_zeroes_percent,
            min_unused_hi_bits_percent: flags.min_unused_hi_bits_percent,
        },
    );

    // Extract system properties first: java.vm.inputarguments (when present)
    // is the authoritative source for JVM flags, and the G1 region size is
    // needed by the linear scan's humongous-object detection.
    let sys_props = coordinator.extract_system_properties(heap_dump_path);
    extract_jvm_info(
        heap_arc.as_ref(),
        stats_arc.as_ref(),
        flags.jvm_flags_content.as_deref(),
        &sys_props,
    );

    coordinator.run(heap_dump_path)
        .with_context(|| format!("Linear scan failed on {}", heap_dump_path.display()))?;

    // Extract thread names by walking Thread instances
    let thread_info = coordinator.extract_thread_info(heap_dump_path);

    // Determine OOM thread serial and extract local variable details
    {
        let idle_methods: &[&str] = &["wait", "park", "parkNanos", "parkUntil", "sleep", "yield", "waitForReferencePendingList"];
        let mut oom_serial: Option<u32> = None;
        let mut oom_depth: usize = 0;
        for (&serial, st) in &heap_arc.stack_traces {
            if let Some(first_frame) = st.frames.first() {
                if first_frame.line_number == -3 {
                    let is_idle = idle_methods.iter().any(|m| first_frame.method_name == *m);
                    if !is_idle && st.frames.len() > oom_depth {
                        oom_serial = Some(serial);
                        oom_depth = st.frames.len();
                    }
                }
            }
        }
        if let Some(serial) = oom_serial {
            let locals = coordinator.extract_oom_local_details(heap_dump_path, serial);
            let mut ol = stats_arc.oom_frame_locals.lock().expect("stats mutex poisoned");
            *ol = locals;
        }
    }

    drop(coordinator);

    // Store thread info in StatsEngine
    {
        let mut ti = stats_arc.thread_info.lock().expect("stats mutex poisoned");
        *ti = thread_info;
    }
    print_elapsed("Phase 3 (linear scan + extraction)", t.elapsed());
    ss_support::mem_trace::trace("phase3-linear-scan");

    // Phase 3b: BFS traversal from GC roots
    eprintln!("\n=== Phase 3b: BFS Traversal ===");
    let t = Instant::now();
    let traverser = ss_traversal::heap_traverser::HeapObjectTraverser::new(
        Arc::clone(&heap_arc),
        Arc::clone(&stats_arc),
    );
    let traversal_result = traverser.run_traversal(heap_dump_path)
        .with_context(|| format!("BFS traversal failed on {}", heap_dump_path.display()))?;
    eprintln!(
        "  Traversal: {} live, {} garbage ({:.1}%), total shallow: {}",
        traversal_result.visited_count,
        traversal_result.garbage_count,
        if traversal_result.total_objects > 0 {
            traversal_result.garbage_count as f64 / traversal_result.total_objects as f64 * 100.0
        } else { 0.0 },
        ss_report::table_formatter::format_bytes(traversal_result.total_shallow),
    );
    drop(traverser);
    print_elapsed("Phase 3b", t.elapsed());
    ss_support::mem_trace::trace("phase3b-bfs");

    // Opt-in S14 forensics: join the detector's duplicate groups with the
    // final BFS live set. Inert when SS_MAT_DEBUG_DIR is not set.
    if let Some(debug_dir) = ss_stats::debug_dump::debug_dir() {
        let live_prim = {
            let guard = stats_arc.traversal_data.lock().expect("stats mutex poisoned");
            guard
                .as_ref()
                .map(ss_stats::debug_dump::live_primitive_array_ids)
                .unwrap_or_default()
        };
        match ss_stats::debug_dump::write_s14_bundle(&debug_dir, &live_prim) {
            Ok(s) => eprintln!(
                "Debug: S14 groups={} arrays={} waste={}B live={} garbage={} string_backing={} -> {}",
                s.groups,
                s.arrays_in_groups,
                s.waste_bytes,
                s.live_members,
                s.garbage_members,
                s.string_backing_members,
                debug_dir.display()
            ),
            Err(e) => return Err(e).context("write requested S14 debug dump"),
        }
        match ss_stats::debug_dump::write_problem_bundle(&debug_dir, stats_arc.as_ref(), heap_arc.as_ref()) {
            Ok(s) => eprintln!(
                "Debug: problem dump S8 {}/{} dup={} {}B groups={} | S9 objects={} bad={} {}B | S10 objects={} bad={} {}B | S11 objects={} bad={} {}B -> {}",
                s.s8_total_strings,
                s.s8_unique_strings,
                s.s8_duplicate_values,
                s.s8_wasted_bytes,
                s.s8_groups_dumped,
                s.collection_total_objects,
                s.collection_problem_count,
                s.collection_overhead,
                s.obj_array_total_objects,
                s.obj_array_problem_count,
                s.obj_array_overhead,
                s.prim_array_total_objects,
                s.prim_array_problem_count,
                s.prim_array_overhead,
                debug_dir.display()
            ),
            Err(e) => return Err(e).context("write requested problem debug dump"),
        }
    }

    // Phase 4: Generate HTML report
    eprintln!("\n=== Phase 4: Generating Report ===");
    let t = Instant::now();
    let mut reporter = ss_report::main_report::MainReportGenerator::new(
        Arc::clone(&heap_arc),
        Arc::clone(&stats_arc),
        heap_dump_path.file_name().and_then(|n| n.to_str()).unwrap_or("heap_dump").to_string(),
    )
    .with_hprof_path(heap_dump_path);
    let html = reporter.try_generate().context("generate report from result stores")?;

    publish_report(&report_path, html.as_bytes())
        .with_context(|| format!("Failed to write report to {}", report_path.display()))?;

    eprintln!("Report written to: {}", report_path.display());
    print_elapsed("Phase 4", t.elapsed());
    ss_support::mem_trace::trace("phase4-report");

    // Print final summary
    if let Some(stats) = Arc::into_inner(stats_arc) {
        stats.print_summary();
    }

    ss_support::mem_trace::trace("end");
    ss_support::mem_trace::summary();
    eprintln!("\nAnalysis complete. Total time: {:.2}s", run_start.elapsed().as_secs_f64());
    Ok(())
}

/// Turn a raw HPROF parse error into a clear, actionable message.
fn friendly_hprof_error(e: &anyhow::Error, path: &PathBuf) -> anyhow::Error {
    let msg = format!("{:?}", e);
    // Detect common failure signatures.
    if msg.contains("TakeUntil") || msg.contains("Eof") {
        anyhow::anyhow!(
            "The file '{}' is not a valid HPROF heap dump, or is truncated/corrupted.\n\
             HPROF files start with the bytes \"JAVA PROFILE\"; ensure you passed a .hprof\n\
             produced by -XX:+HeapDumpOnOutOfMemoryError or jmap.\n\
             Underlying parser error: {}",
            path.display(), e
        )
    } else if msg.contains("Unsupported HPROF version") {
        anyhow::anyhow!(
            "Unsupported HPROF version in '{}'. SS-Mat supports HPROF versions 1.0.2 and 1.0.3.\n\
             Underlying error: {}",
            path.display(), e
        )
    } else {
        anyhow::anyhow!(
            "Failed to parse HPROF file '{}': {}\n\
             The file may be corrupted or use an unsupported format.",
            path.display(), e
        )
    }
}

fn print_elapsed(label: &str, elapsed: std::time::Duration) {
    eprintln!("  {} took {:.2}s", label, elapsed.as_secs_f64());
}

fn populate_per_class_stats(
    heap: &ss_heap_model::heap_dump::HeapDump,
    stats: &ss_stats::engine::StatsEngine,
) {
    use std::sync::atomic::Ordering::Relaxed;

    for (i, cls) in heap.classes.iter().enumerate() {
        let idx = cls.index;
        if let Some(pc) = stats.get_per_class(idx) {
            let count = heap.per_class_inst_count.get(i).copied().unwrap_or(0);
            let bytes = heap.per_class_inst_bytes.get(i).copied().unwrap_or(0);
            pc.instance_count.store(count, Relaxed);
            pc.total_instance_bytes.store(bytes, Relaxed);
        }
    }

    stats
        .total_instances
        .store(heap.total_object_count, Relaxed);
}

fn validate_inputs(path: &PathBuf) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("File not found: {}", path.display());
    }
    if !path.is_file() {
        anyhow::bail!("Not a regular file: {}", path.display());
    }

    // Warn on unusually small files (likely not a real heap dump).
    if let Ok(meta) = path.metadata() {
        if meta.len() < 1024 {
            anyhow::bail!(
                "File {} is only {} bytes — too small to be a valid HPROF heap dump.",
                path.display(), meta.len()
            );
        }
    }

    let compressed_extensions = [".zip", ".gz", ".tgz", ".rar", ".bz2"];
    let name = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    let name_lower = name.to_lowercase();
    for ext in &compressed_extensions {
        if name_lower.ends_with(ext) {
            anyhow::bail!(
                "Compressed file detected ({}). Please decompress the heap dump first.",
                ext
            );
        }
    }

    Ok(())
}

fn derive_report_path(heap_dump: &PathBuf) -> PathBuf {
    let stem = heap_dump
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("report");
    heap_dump.with_file_name(format!("{}.html", stem))
}

fn extract_jvm_info(
    heap: &ss_heap_model::heap_dump::HeapDump,
    stats: &ss_stats::engine::StatsEngine,
    jvm_flags_file: Option<&str>,
    sys_props: &[(String, String)],
) {
    // JVM startup flags are not stored in HPROF by default. Use, in order:
    //   1. an explicit --jvm-flags-file
    //   2. java.vm.inputarguments / jxray.java.vm.inputarguments properties
    // Never guess flags by scanning arbitrary heap strings.
    let mut jvm_flags_raw = jvm_flags_file.unwrap_or("").trim().to_string();
    if jvm_flags_raw.is_empty() {
        jvm_flags_raw = sys_props
            .iter()
            .find(|(k, _)| k == "java.vm.inputarguments" || k == "jxray.java.vm.inputarguments")
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
    }

    let mut gc_type = String::from("Unknown");

    // Detect GC type from flags (iterate in reverse: last flag wins)
    let flags: Vec<&str> = jvm_flags_raw.split_whitespace().collect();
    for flag in flags.iter().rev() {
        if flag.contains("UseSerialGC") {
            gc_type = "Serial".to_string();
            break;
        } else if flag.contains("UseParallelGC") {
            gc_type = "Parallel".to_string();
            break;
        } else if flag.contains("UseConcMarkSweepGC") {
            gc_type = "CMS".to_string();
            break;
        } else if flag.contains("UseG1GC") {
            gc_type = "G1".to_string();
            break;
        } else if flag.contains("UseZGC") {
            gc_type = "ZGC".to_string();
            break;
        } else if flag.contains("UseShenandoahGC") {
            gc_type = "Shenandoah".to_string();
            break;
        }
    }

    // Parse Xmx from flags
    let mut xmx_value: i64 = -1;
    for flag in &flags {
        if flag.starts_with("-Xmx") {
            xmx_value = parse_memory_flag(&flag[4..]);
        }
    }

    // Parse G1HeapRegionSize from flags
    let mut g1_region_size: i64 = -1;
    for flag in &flags {
        if flag.starts_with("-XX:G1HeapRegionSize=") {
            g1_region_size = parse_memory_flag(&flag[21..]);
        }
    }

    // Do not guess the GC type from the Java version when no flags are
    // available: the report must say "not provided" rather than invent one.

    // Calculate G1 region size if not explicitly set
    if g1_region_size <= 0 {
        let is_64bit = heap.identifier_size >= 8;
        let effective_heap = if xmx_value > 0 {
            xmx_value
        } else {
            let total_bytes = heap.total_object_count as i64 * 64; // rough estimate
            if is_64bit { total_bytes } else { total_bytes * 3 / 2 }
        };
        let heap_mb = effective_heap / (1024 * 1024);
        let divisor = if heap.identifier_size < 15 { 2048 } else { 1024 };
        let target = (heap_mb / divisor).max(1);
        // Round up to next power of 2
        let mut region_mb = 1i64;
        while region_mb < target { region_mb *= 2; }
        region_mb = region_mb.max(1).min(32);
        if xmx_value <= 0 && region_mb == 1 { region_mb = 2; }
        g1_region_size = region_mb * 1024 * 1024;
    }

    // Prefer the real System.getProperties() object graph. The string-table
    // heuristics below only fill keys that the object graph did not provide.
    let mut value_map: std::collections::HashMap<String, String> =
        sys_props.iter().cloned().collect();

    for (_id, s) in &heap.string_table {
        let s = s.as_str();

        // Runtime name
        if (s.starts_with("Java(TM) SE Runtime Environment") || s.starts_with("OpenJDK Runtime"))
            && !value_map.contains_key("java.runtime.name")
        {
            value_map.insert("java.runtime.name".to_string(), s.to_string());
        }
        // Java version
        else if (s.starts_with("1.8.") || s.starts_with("11.") || s.starts_with("17.")
            || s.starts_with("21.") || s.starts_with("22.") || s.starts_with("23.")
            || s.starts_with("24.") || s.starts_with("25."))
            && s.len() >= 5 && s.len() < 30
            && !value_map.contains_key("java.version")
        {
            value_map.insert("java.version".to_string(), s.to_string());
        }
        // JVM name
        else if (s.starts_with("Java HotSpot") || s.starts_with("OpenJDK"))
            && s.len() < 80
            && !value_map.contains_key("java.vm.name")
        {
            value_map.insert("java.vm.name".to_string(), s.to_string());
        }
        // OS name
        else if (s == "Linux" || s == "Mac OS X" || s.starts_with("Windows"))
            && !value_map.contains_key("os.name")
        {
            value_map.insert("os.name".to_string(), s.to_string());
        }
        // OS arch
        else if (s == "amd64" || s == "x86_64" || s == "aarch64" || s == "ppc64le" || s == "s390x")
            && !value_map.contains_key("os.arch")
        {
            value_map.insert("os.arch".to_string(), s.to_string());
        }
        // File encoding
        else if (s == "UTF-8" || s == "ISO-8859-1" || s == "ASCII" || s == "GBK" || s.starts_with("ISO-8859"))
            && s.len() < 20
            && !value_map.contains_key("file.encoding")
        {
            value_map.insert("file.encoding".to_string(), s.to_string());
        }
        // java.home - paths starting with /usr/lib/jvm or similar
        else if (s.starts_with("/usr/lib/jvm") || s.starts_with("/Library/Java/") || s.starts_with("C:\\Program Files\\Java"))
            && s.len() < 200 && !s.contains('\0')
            && !value_map.contains_key("java.home")
        {
            value_map.insert("java.home".to_string(), s.to_string());
        }
        // user.dir / user.home - common path patterns
        else if s.starts_with("/home/") && s.len() < 100 && s.matches('/').count() <= 4
            && !value_map.contains_key("user.home")
        {
            value_map.insert("user.home".to_string(), s.to_string());
        }
        else if s.starts_with("/opt/") && s.len() < 100 && s.matches('/').count() <= 4
            && !value_map.contains_key("user.dir")
            && !s.contains(".jar") && !s.contains(".log")
        {
            value_map.insert("user.dir".to_string(), s.to_string());
        }
        // OS version (e.g. "5.4.0-150-generic", "10.0", "12.0")
        else if (s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
            && s.contains('.') && s.len() < 50 && !s.contains("http") && !s.contains(".class")
            && !s.contains("/") && !s.contains("\\")
            && !value_map.contains_key("os.version"))
        {
            // Heuristic: short digit-starting string with dots could be OS version
            // Only accept if we already found os.name
            if value_map.contains_key("os.name") {
                value_map.insert("os.version".to_string(), s.to_string());
            }
        }
    }

    let ptr_size = heap.identifier_size;

    // Build sorted property list
    let mut sorted_props: Vec<(String, String)> = value_map.into_iter().collect();
    sorted_props.sort_by(|a, b| a.0.cmp(&b.0));
    sorted_props.dedup_by(|a, b| a.0 == b.0);

    let info = ss_stats::jvm_info::JvmInfo {
        address_size: ptr_size as i32 * 8,
        id_size: ptr_size as i32,
        gc_type,
        xmx: xmx_value.max(0),
        xms: 0,
        jvm_flags_raw,
        system_properties: sorted_props,
        g1_region_size,
    };

    stats.jvm_info.set(info);
}

/// Removes the files this run wrote into its temp workspace.
///
/// The object directory and the BFS spill files are the largest thing a run
/// writes outside the report, and the workspace is only removed by its own guard
/// when this run created it: with `--temp-dir` pointing at a directory that
/// already existed, the artifacts stayed behind. The guard runs on the error path
/// too, because `main` returns `Result` and the drop happens on the way out.
struct TempFiles {
    path: PathBuf,
    /// Only this run's files: the names carry the pid, so a concurrent run
    /// sharing the directory keeps its own.
    pid_tag: String,
}

impl TempFiles {
    fn new(path: PathBuf) -> Self {
        Self { path, pid_tag: format!("-{}", std::process::id()) }
    }
}

impl Drop for TempFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.path.join("objects.objdir"));
        let Ok(entries) = std::fs::read_dir(&self.path) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !(name.starts_with("bfs-") || name.starts_with("ss-mat-scan-")) {
                continue;
            }
            if !name.contains(&self.pid_tag) {
                continue;
            }
            let path = entry.path();
            let _ = if path.is_dir() {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
        }
    }
}

/// Parse a memory flag value like "512m", "2g", "1024k" into bytes.
fn parse_memory_flag(s: &str) -> i64 {
    let s = s.trim();
    if s.is_empty() { return -1; }
    let (num_part, multiplier) = if s.ends_with('g') || s.ends_with('G') {
        (&s[..s.len()-1], 1024i64 * 1024 * 1024)
    } else if s.ends_with('m') || s.ends_with('M') {
        (&s[..s.len()-1], 1024i64 * 1024)
    } else if s.ends_with('k') || s.ends_with('K') {
        (&s[..s.len()-1], 1024i64)
    } else {
        (s, 1i64)
    };
    match num_part.parse::<i64>() {
        Ok(v) => v * multiplier,
        Err(_) => -1,
    }
}

fn publish_report(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| std::path::Path::new("."));
    let (temporary, mut file) = loop {
        let temporary = parent.join(format!(".ss-mat-report-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&temporary) {
            Ok(file) => break (temporary, file),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let result = (|| { file.write_all(bytes)?; file.sync_all()?; std::fs::rename(&temporary, path) })();
    if result.is_err() { let _ = std::fs::remove_file(&temporary); }
    result
}

#[cfg(test)]
mod publication_tests {
    #[test]
    fn failed_publication_does_not_replace_destination() {
        let base=std::env::temp_dir().join(format!("ss-mat-publication-test-{}",std::process::id()));
        std::fs::create_dir(&base).unwrap();
        let target=base.join("destination");std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("sentinel"),b"original").unwrap();
        assert!(super::publish_report(&target,b"new").is_err());
        assert_eq!(std::fs::read(target.join("sentinel")).unwrap(),b"original");
        assert_eq!(std::fs::read_dir(&base).unwrap().count(),1);
        std::fs::remove_dir_all(base).unwrap();
    }
}
