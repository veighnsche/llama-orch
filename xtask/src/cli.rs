use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "xtask", version, about = "Workspace utility tasks")]
pub struct Xtask {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    #[command(name = "regen-openapi")]
    RegenOpenapi,
    #[command(name = "regen-schema")]
    RegenSchema,
    #[command(name = "regen")]
    Regen,
    #[command(name = "spec-extract")]
    SpecExtract,
    #[command(name = "dev:loop")]
    DevLoop,
    #[command(name = "ci:haiku:cpu")]
    CiHaikuCpu,
    #[command(name = "ci:determinism")]
    CiDeterminism,
    #[command(name = "ci:auth")]
    CiAuth,
    #[command(name = "pact:verify")]
    PactVerify,
    #[command(name = "docs:index")]
    DocsIndex,
    #[command(name = "engine:status")]
    EngineStatus {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to check (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
    #[command(name = "engine:down")]
    EngineDown {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to stop (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
    #[command(name = "bdd:test")]
    BddTest {
        /// Run tests with specific tag (e.g., @auth, @p0)
        #[arg(long)]
        tags: Option<String>,
        /// Run specific feature file (e.g., lifecycle, authentication)
        #[arg(long)]
        feature: Option<String>,
        /// DEPRECATED: Use --really-quiet instead. This flag now shows a warning.
        #[arg(long, short)]
        quiet: bool,
        /// Actually suppress live output (only show summary). Use this for CI/CD.
        #[arg(long)]
        really_quiet: bool,
        /// Run ALL tests (default: only failing tests from last run)
        #[arg(long)]
        all: bool,
    },
    #[command(name = "bdd:tail")]
    BddTail {
        /// Number of lines to show (default: 50)
        #[arg(short, long, default_value = "50")]
        lines: usize,
    },
    #[command(name = "bdd:head")]
    BddHead {
        /// Number of lines to show (default: 100)
        #[arg(short, long, default_value = "100")]
        lines: usize,
    },
    #[command(name = "bdd:grep")]
    BddGrep {
        /// Pattern to search for
        pattern: String,
        /// Case insensitive search
        #[arg(short, long)]
        ignore_case: bool,
    },
    #[command(name = "bdd:check-duplicates")]
    BddCheckDuplicates,
    #[command(name = "bdd:fix-duplicates")]
    BddFixDuplicates,
    #[command(name = "bdd:analyze")]
    BddAnalyze {
        /// Show detailed file-by-file breakdown
        #[arg(long)]
        detailed: bool,
        /// Show only files with stubs
        #[arg(long)]
        stubs_only: bool,
        /// Output format (text, json, markdown)
        #[arg(long, default_value = "text")]
        format: String,
    },
    #[command(name = "bdd:progress")]
    BddProgress {
        /// Compare with previous run (requires .bdd-progress.json)
        #[arg(long)]
        compare: bool,
    },
    #[command(name = "bdd:stubs")]
    BddStubs {
        /// Show stubs for specific file
        #[arg(long)]
        file: Option<String>,
        /// Minimum stub count to show (default: 1)
        #[arg(long, default_value = "1")]
        min_stubs: usize,
    },
    #[command(name = "worker:test")]
    WorkerTest {
        /// Worker ID (default: auto-generated UUID)
        #[arg(long)]
        worker_id: Option<String>,
        /// Model path (default: ../../.test-models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
        #[arg(long)]
        model: Option<PathBuf>,
        /// Backend (cpu, cuda, metal)
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Device ID
        #[arg(long, default_value = "0")]
        device: u32,
        /// Worker port
        #[arg(long, default_value = "18081")]
        port: u16,
        /// Mock hive port
        #[arg(long, default_value = "19200")]
        hive_port: u16,
        /// Timeout in seconds
        #[arg(long, default_value = "30")]
        timeout: u64,
    },
    /// TEAM-160: E2E test - Queen lifecycle (start/stop)
    #[command(name = "e2e:queen")]
    E2eQueen,

    /// TEAM-160: E2E test - Hive lifecycle (start/stop)
    #[command(name = "e2e:hive")]
    E2eHive,

    /// TEAM-160: E2E test - Cascade shutdown (queen → hive)
    #[command(name = "e2e:cascade")]
    E2eCascade,

    /// Smart wrapper for rbee-keeper: auto-builds if needed, then forwards command
    #[command(name = "rbee", trailing_var_arg = true, allow_hyphen_values = true)]
    Rbee {
        /// Arguments to forward to rbee-keeper
        #[arg(allow_hyphen_values = true)]
        args: Vec<String>,
    },
}
