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
    #[command(name = "ci:haiku:gpu")]
    CiHaikuGpu,
    #[command(name = "ci:haiku:cpu")]
    CiHaikuCpu,
    #[command(name = "ci:determinism")]
    CiDeterminism,
    #[command(name = "ci:auth")]
    CiAuth,
    #[command(name = "pact:verify")]
    PactVerify,
    #[command(name = "pact:publish")]
    PactPublish,
    #[command(name = "docs:index")]
    DocsIndex,
    #[command(name = "engine:plan")]
    EnginePlan {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to plan for (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
    #[command(name = "engine:up")]
    EngineUp {
        /// Path to config YAML
        #[arg(long, default_value = "requirements/llamacpp-3090-source.yaml")]
        config: PathBuf,
        /// Optional pool id to provision (defaults to all)
        #[arg(long)]
        pool: Option<String>,
    },
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
}
