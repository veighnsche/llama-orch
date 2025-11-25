use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", version, about = "Workspace utility tasks")]
pub struct Xtask {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    // TEAM-451: Release management
    // TEAM-452: Removed tier system - now app-based
    // TEAM-XXX: Added --app flag for non-interactive usage
    #[command(name = "release")]
    Release {
        /// App to release (gwc, commercial, marketplace, docs, keeper, queen, hive)
        #[arg(long)]
        app: Option<String>,
        /// Bump type (patch, minor, major)
        #[arg(long)]
        r#type: Option<String>,
        /// Dry run (preview changes without applying)
        #[arg(long)]
        dry_run: bool,
        /// CI mode - skip confirmation prompts
        #[arg(long)]
        ci: bool,
    },
    // TEAM-451: Cloudflare deployment
    // TEAM-463: Added --production flag for production deployments
    #[command(name = "deploy")]
    Deploy {
        /// App to deploy (worker, commercial, marketplace, docs)
        #[arg(long)]
        app: String,
        /// Version bump type (patch, minor, major) - bumps version before deploying
        #[arg(long)]
        bump: Option<String>,
        /// Deploy to production (default: preview)
        #[arg(long)]
        production: bool,
        /// Dry run (preview commands without executing)
        #[arg(long)]
        dry_run: bool,
    },
    /// TEAM_531: Interactive Turbo dev scope selector (wraps `turbo dev` with presets)
    #[command(name = "dev-scope")]
    DevScope {
        /// Print discovered packages & presets, do not start TUI or turbo
        #[arg(long)]
        print_only: bool,
    },
    /// Smart wrapper for rbee-keeper: auto-builds if needed, then forwards command
    #[command(name = "rbee", trailing_var_arg = true, allow_hyphen_values = true)]
    Rbee {
        /// Arguments to forward to rbee-keeper
        #[arg(allow_hyphen_values = true)]
        args: Vec<String>,
    },
}
