mod flags;
mod progress;

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<()> {
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

    let _flags = flags::Flags::from_cli(&cli);

    eprintln!("Heap dump: {}", heap_dump_path.display());
    eprintln!("Report output: {}", report_path.display());

    // Phase 1: Parse HPROF file
    eprintln!("\n=== Phase 1: Parsing HPROF ===");
    let heap = ss_hprof_parser::hprof_wrapper::parse_hprof(heap_dump_path)
        .context("Failed to parse HPROF file")?;

    eprintln!(
        "  Classes: {}, Objects: {}, Roots: {}, Strings: {}",
        heap.classes.len(),
        heap.total_object_count,
        heap.roots.len(),
        heap.string_table.len()
    );

    // Phase 2: Initialize StatsEngine with per-class data
    eprintln!("\n=== Phase 2: Statistics Engine ===");
    let stats = ss_stats::engine::StatsEngine::new(&heap.classes);
    populate_per_class_stats(&heap, &stats);

    // Extract JVM info from string table
    extract_jvm_info(&heap, &stats);

    // Phase 3: Linear scan
    eprintln!("\n=== Phase 3: Linear Scan ===");
    let heap_arc = Arc::new(heap);
    let stats_arc = Arc::new(stats);
    let coordinator =
        ss_traversal::linear_scan::LinearScanCoordinator::new(Arc::clone(&heap_arc), Arc::clone(&stats_arc));
    coordinator.run(heap_dump_path)?;

    // Extract system properties by walking java.lang.System.props object graph
    let sys_props = coordinator.extract_system_properties(heap_dump_path);

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
            let mut ol = stats_arc.oom_frame_locals.lock().unwrap();
            *ol = locals;
        }
    }

    drop(coordinator);

    // Update JVM info with extracted system properties
    update_system_properties(&stats_arc, sys_props);

    // Store thread info in StatsEngine
    {
        let mut ti = stats_arc.thread_info.lock().unwrap();
        *ti = thread_info;
    }

    // Phase 3b: BFS traversal from GC roots
    eprintln!("\n=== Phase 3b: BFS Traversal ===");
    let traverser = ss_traversal::heap_traverser::HeapObjectTraverser::new(
        Arc::clone(&heap_arc),
        Arc::clone(&stats_arc),
    );
    let traversal_result = traverser.run_traversal(heap_dump_path)?;
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

    // Phase 4: Generate HTML report
    eprintln!("\n=== Phase 4: Generating Report ===");
    let mut reporter = ss_report::main_report::MainReportGenerator::new(
        Arc::clone(&heap_arc),
        Arc::clone(&stats_arc),
        heap_dump_path.file_name().and_then(|n| n.to_str()).unwrap_or("heap_dump").to_string(),
    );
    let html = reporter.generate();

    std::fs::write(&report_path, &html)
        .with_context(|| format!("Failed to write report to {}", report_path.display()))?;

    eprintln!("Report written to: {}", report_path.display());

    // Print final summary
    if let Some(stats) = Arc::into_inner(stats_arc) {
        stats.print_summary();
    }

    eprintln!("\nAnalysis complete.");
    Ok(())
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

fn extract_jvm_info(heap: &ss_heap_model::heap_dump::HeapDump, stats: &ss_stats::engine::StatsEngine) {
    let mut jvm_flags_raw = String::new();
    let mut gc_type = String::from("Unknown");

    // Collect JVM flags with strict filtering to avoid garbled output
    for (_id, s) in &heap.string_table {
        if (s.starts_with("-X") || s.starts_with("-XX:")) && s.len() < 200 && !s.contains('\0') && !s.contains('\n') {
            // Must look like a real JVM flag: starts with known prefix + reasonable length
            let is_valid = if s.starts_with("-Xmx") || s.starts_with("-Xms") || s.starts_with("-Xmn") || s.starts_with("-Xss") {
                true // -Xmx<size>, -Xms<size> etc.
            } else if s.starts_with("-XX:") {
                // -XX:+Flag, -XX:-Flag, -XX:Flag=value
                s.contains('=') || s.contains("+") || s.contains("-")
            } else if s.starts_with("-X") {
                s.len() < 50 // Other -X flags should be short
            } else {
                false
            };
            if is_valid {
                if !jvm_flags_raw.is_empty() { jvm_flags_raw.push(' '); }
                jvm_flags_raw.push_str(s);
            }
        }
    }

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

    // Default GC type based on Java version if no explicit flag found
    if gc_type == "Unknown" {
        let java_version = heap.string_table.values().find(|s| {
            (s.starts_with("1.8.") || s.starts_with("9.") || s.starts_with("1")
                || (s.len() >= 5 && s.len() < 30
                    && s.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false)
                    && s.contains('.')
                    && s.chars().take(2).all(|c| c.is_ascii_digit())))
                && !s.starts_with("-")
        }).and_then(|s| {
            // Extract major version
            if s.starts_with("1.8.") { Some(8) }
            else if let Some(dot) = s.find('.') {
                s[..dot].parse::<i32>().ok()
            } else { None }
        });

        gc_type = if java_version.unwrap_or(8) >= 9 {
            "G1".to_string()
        } else {
            "Parallel".to_string()
        };
    }

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

    // Extract system property values by pattern matching
    let mut value_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();

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

    // Also try to detect pointer size from os.arch
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

/// Update JVM info with system properties extracted from the object graph.
fn update_system_properties(stats: &Arc<ss_stats::engine::StatsEngine>, props: Vec<(String, String)>) {
    if props.is_empty() {
        return;
    }
    let mut info = stats.jvm_info.get();
    info.system_properties = props;
    stats.jvm_info.set(info);
}
