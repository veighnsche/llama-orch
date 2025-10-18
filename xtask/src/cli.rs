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
        /// Suppress live output (only show summary)
        #[arg(long, short)]
        quiet: bool,
    },
}
