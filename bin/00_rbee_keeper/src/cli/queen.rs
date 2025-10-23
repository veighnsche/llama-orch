//! Queen-rbee CLI actions
//!
//! TEAM-276: Extracted from main.rs

use clap::Subcommand;

#[derive(Subcommand)]
pub enum QueenAction {
    /// Start queen-rbee daemon
    Start,
    /// Stop queen-rbee daemon
    Stop,
    /// Check queen-rbee daemon status
    Status,
    /// Rebuild queen with different configuration
    /// TEAM-262: Added for local-hive optimization
    Rebuild {
        /// Include local hive for localhost operations (50-100x faster)
        #[arg(long)]
        with_local_hive: bool,
    },
    /// Show queen build configuration
    /// TEAM-262: Query /v1/build-info endpoint
    Info,
    /// Install queen binary
    /// TEAM-262: Similar to hive install
    Install {
        /// Binary path (optional, auto-detect from target/)
        #[arg(short, long)]
        binary: Option<String>,
    },
    /// Uninstall queen binary
    /// TEAM-262: Similar to hive uninstall
    Uninstall,
}
