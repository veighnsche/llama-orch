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
}
