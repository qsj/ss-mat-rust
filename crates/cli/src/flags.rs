use clap::Parser;
use std::path::PathBuf;

/// SS-Mat: JVM heap dump analysis tool (Rust version)
#[derive(Parser, Debug)]
#[command(name = "ss-mat", version = "0.1.0")]
#[command(about = "Analyze JVM heap dumps and generate HTML reports")]
pub struct Cli {
    /// Path to the HPROF heap dump file
    pub heap_dump: Option<PathBuf>,

    /// Path for the output HTML report (default: <heap_dump>.html)
    pub report_output: Option<PathBuf>,

    // --- Analysis thresholds ---
    /// Minimum trailing zeroes percentage to flag primitive arrays (default: 25.0)
    #[arg(long, default_value = "25.0")]
    pub min_trailing_zeroes_percent: f64,

    /// Minimum sparse zeroes percentage to flag arrays (default: 70.0)
    #[arg(long, default_value = "70.0")]
    pub min_sparse_zeroes_percent: f64,

    /// Minimum unused high bits percentage to flag arrays (default: 70.0)
    #[arg(long, default_value = "70.0")]
    pub min_unused_hi_bits_percent: f64,

    // --- JVM info ---
    /// Path to file containing JVM startup flags (one per line)
    #[arg(long)]
    pub jvm_flags_file: Option<PathBuf>,

    // --- Debug options ---
    /// Maximum number of object graph scan threads (0 = auto)
    #[arg(long, default_value_t = 0)]
    pub xdebug_max_obj_graph_scan_threads: i32,

}

/// Global configuration flags, mirroring Java's Flags class.
pub struct Flags {
    pub version: &'static str,
    pub min_trailing_zeroes_percent: f64,
    pub min_sparse_zeroes_percent: f64,
    pub min_unused_hi_bits_percent: f64,
    pub max_scan_threads: i32,
    pub jvm_flags_content: Option<String>,
}

impl Flags {
    pub fn from_cli(cli: &Cli) -> Self {
        let jvm_flags_content = cli
            .jvm_flags_file
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok());

        Self {
            version: "0.1.0",
            min_trailing_zeroes_percent: cli.min_trailing_zeroes_percent,
            min_sparse_zeroes_percent: cli.min_sparse_zeroes_percent,
            min_unused_hi_bits_percent: cli.min_unused_hi_bits_percent,
            max_scan_threads: cli.xdebug_max_obj_graph_scan_threads,
            jvm_flags_content,
        }
    }

}

pub fn print_help() {
    println!("SS-Mat - JVM Heap Dump Analysis Tool (Rust version)");
    println!();
    println!("Usage: ss-mat <heap_dump.hprof> [report.html] [options]");
    println!();
    println!("Options:");
    println!("  --min_trailing_zeroes_percent <f64>  Min trailing zeroes % (default: 25.0)");
    println!("  --min_sparse_zeroes_percent <f64>  Min sparse zeroes % (default: 70.0)");
    println!("  --min_unused_hi_bits_percent <f64> Min unused high bits % (default: 70.0)");
    println!("  --jvm_flags_file <path>           JVM flags file path");
    println!("  --xdebug_max_obj_graph_scan_threads <int>  Max scan threads (default: auto)");
    println!("  -h, --help                        Print this help");
}
