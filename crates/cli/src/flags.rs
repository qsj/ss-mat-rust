use clap::Parser;
use std::path::PathBuf;

/// SS-Mat: JVM heap dump analysis tool (Rust version)
#[derive(Parser, Debug)]
#[command(name = "ss-mat", version = "0.1.0")]
#[command(about = "Analyze JVM heap dumps and generate HTML reports")]
#[command(disable_help_flag = true)]
pub struct Cli {
    /// Path to the HPROF heap dump file
    pub heap_dump: Option<PathBuf>,

    /// Path for the output HTML report (default: <heap_dump>.html)
    pub report_output: Option<PathBuf>,

    // --- Analysis thresholds ---
    /// Minimum percentage of heap for reporting problems (default: 0.1)
    #[arg(long, default_value = "0.1")]
    pub min_reported_percent: f64,

    /// Minimum trailing zeroes percentage to flag primitive arrays (default: 25.0)
    #[arg(long, default_value = "25.0")]
    pub min_trailing_zeroes_percent: f64,

    /// Minimum sparse zeroes percentage to flag arrays (default: 70.0)
    #[arg(long, default_value = "70.0")]
    pub min_sparse_zeroes_percent: f64,

    /// Minimum unused high bits percentage to flag arrays (default: 70.0)
    #[arg(long, default_value = "70.0")]
    pub min_unused_hi_bits_percent: f64,

    /// Minimum duplicate object values to print (default: 20)
    #[arg(long, default_value_t = 20)]
    pub min_dup_obj_vals_to_print: i32,

    // --- HPROF parser options ---
    /// Force HPROF pointer size (4 or 8 bytes, 0 = auto-detect)
    #[arg(long, default_value_t = 0)]
    pub hprof_pointer_size: i32,

    /// HPROF file cache page size bits
    #[arg(long, default_value_t = 0)]
    pub hprof_file_cache_page_size_bits: i32,

    /// HPROF file cache size in bytes (0 = auto)
    #[arg(long, default_value_t = 0)]
    pub hprof_file_cache_size: i64,

    /// Disable file size check against available memory
    #[arg(long, default_value_t = false)]
    pub disable_file_size_check: bool,

    // --- JVM info ---
    /// Path to file containing JVM startup flags (one per line)
    #[arg(long)]
    pub jvm_flags_file: Option<PathBuf>,

    // --- Debug options ---
    /// Maximum number of object graph scan threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    pub xdebug_max_obj_graph_scan_threads: i32,

    /// Extra class filters (comma-separated class names)
    #[arg(long)]
    pub extra_classes: Option<String>,
}

/// Global configuration flags, mirroring Java's Flags class.
pub struct Flags {
    pub version: &'static str,
    pub min_reported_percent: f64,
    pub min_trailing_zeroes_percent: f64,
    pub min_sparse_zeroes_percent: f64,
    pub min_unused_hi_bits_percent: f64,
    pub min_dup_obj_vals_to_print: i32,
    pub hprof_pointer_size: i32,
    pub hprof_file_cache_size: i64,
    pub hprof_file_cache_page_size_bits: i32,
    pub file_size_check_enabled: bool,
    pub max_scan_threads: i32,
    pub jvm_flags_content: Option<String>,
    pub extra_class_filters: Option<Vec<String>>,
}

impl Flags {
    pub fn from_cli(cli: &Cli) -> Self {
        let jvm_flags_content = cli
            .jvm_flags_file
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok());

        let extra_class_filters = cli
            .extra_classes
            .as_ref()
            .map(|s| s.split(',').map(|c| c.trim().to_string()).collect());

        let file_size_check_enabled = !cli.disable_file_size_check;

        let hprof_file_cache_size = if !file_size_check_enabled && cli.hprof_file_cache_size == 0 {
            -1
        } else {
            cli.hprof_file_cache_size
        };

        Self {
            version: "0.1.0",
            min_reported_percent: cli.min_reported_percent,
            min_trailing_zeroes_percent: cli.min_trailing_zeroes_percent,
            min_sparse_zeroes_percent: cli.min_sparse_zeroes_percent,
            min_unused_hi_bits_percent: cli.min_unused_hi_bits_percent,
            min_dup_obj_vals_to_print: cli.min_dup_obj_vals_to_print,
            hprof_pointer_size: cli.hprof_pointer_size,
            hprof_file_cache_size,
            hprof_file_cache_page_size_bits: cli.hprof_file_cache_page_size_bits,
            file_size_check_enabled,
            max_scan_threads: cli.xdebug_max_obj_graph_scan_threads,
            jvm_flags_content,
            extra_class_filters,
        }
    }

    /// Calculate the minimum bytes threshold for reporting based on minReportedPercent.
    pub fn calc_percent_of_heap(&self, value: i64) -> i64 {
        ((value as f64 * self.min_reported_percent) / 100.0) as i64
    }

    /// Check if a class name matches any extra class filter.
    pub fn matches_extra_class_filter(&self, class_name: &str) -> bool {
        self.extra_class_filters
            .as_ref()
            .map_or(false, |filters| filters.iter().any(|f| f == class_name))
    }
}

pub fn print_help() {
    println!("SS-Mat - JVM Heap Dump Analysis Tool (Rust version)");
    println!();
    println!("Usage: ss-mat <heap_dump.hprof> [report.html] [options]");
    println!();
    println!("Options:");
    println!("  --min_reported_percent <f64>      Min % of heap for reporting (default: 0.1)");
    println!("  --min_trailing_zeroes_percent <f64>  Min trailing zeroes % (default: 25.0)");
    println!("  --min_sparse_zeroes_percent <f64>  Min sparse zeroes % (default: 70.0)");
    println!("  --min_unused_hi_bits_percent <f64> Min unused high bits % (default: 70.0)");
    println!("  --min_dup_obj_vals_to_print <int>  Min dup values to print (default: 20)");
    println!("  --hprof_pointer_size <int>        Force pointer size 4/8 (default: auto)");
    println!("  --jvm_flags_file <path>           JVM flags file path");
    println!("  --extra_classes <names>           Comma-separated extra class filters");
    println!("  --disable_file_size_check         Disable file size vs memory check");
    println!("  --xdebug_max_obj_graph_scan_threads <int>  Max scan threads (default: auto)");
    println!("  -h, --help                        Print this help");
}
