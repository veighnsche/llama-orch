//! Hive CLI actions (localhost-only)
//!
//! TEAM-322: Removed all remote/SSH functionality (RULE ZERO - delete complexity)

use clap::Subcommand;

#[derive(Subcommand)]
pub enum HiveAction {
    /// Start rbee-hive locally
    Start {
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    /// Stop rbee-hive locally
    Stop {
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port")]
        port: Option<u16>,
    },
    /// Get hive details
    Get {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Check hive status
    Status {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Refresh device capabilities for a hive
    RefreshCapabilities {
        /// Hive alias (only "localhost" supported)
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    /// Run hive-check (narration test through hive SSE)
    Check {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Install rbee-hive binary locally
    Install {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Uninstall rbee-hive binary locally
    Uninstall {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Rebuild rbee-hive binary locally (cargo build)
    Rebuild {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
}
