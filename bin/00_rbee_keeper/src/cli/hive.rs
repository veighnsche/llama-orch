//! Hive CLI actions
//!
//! TEAM-276: Extracted from main.rs

use clap::Subcommand;

#[derive(Subcommand)]
// TEAM-186: Updated to match new install/uninstall workflow
// TEAM-187: Added SshTest for pre-installation SSH validation
// ============================================================
// BUG FIX: TEAM-199 | Clap short option conflict -h vs --help
// ============================================================
// SUSPICION:
// - Error message: "Short option names must be unique for each argument,
//   but '-h' is in use by both 'alias' and 'help'"
// - Suspected all HiveAction variants using #[arg(short = 'h')]
//
// INVESTIGATION:
// - Checked main.rs line 159, 165, 171, 179, 185, 191, 198, 205
// - All HiveAction variants use -h for alias/host parameter
// - Clap auto-generates -h for --help flag
// - Conflict occurs when subcommand is parsed
//
// ROOT CAUSE:
// - Multiple HiveAction variants define #[arg(short = 'h')] for alias
// - Clap reserves -h for --help by default
// - Cannot use same short option for both help and custom argument
//
// FIX:
// - Changed all short = 'h' to short = 'a' (for alias)
// - Long option --host remains unchanged
// - Users can now use -a or --host for alias parameter
// - -h now correctly shows help as expected
//
// TESTING:
// - cargo build --bin rbee-keeper (compilation check)
// - ./rbee hive start (should work without panic)
// - ./rbee hive start -h (should show help)
// - ./rbee hive start -a localhost (should work)
// - ./rbee hive start --host localhost (should work)
// ============================================================
pub enum HiveAction {
    // TEAM-290: RESTORED Install, Uninstall, Start, Stop (now with SSH support via hive-lifecycle)
    /// Install rbee-hive locally or remotely
    Install {
        /// Host to install on (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        host: String,
        /// Path to rbee-hive binary (auto-detects if not specified)
        #[arg(short = 'b', long = "binary")]
        binary: Option<String>,
        /// Installation directory (default: ~/.local/bin)
        #[arg(short = 'd', long = "dir")]
        install_dir: Option<String>,
        /// Build on remote host instead of locally (requires git, slower)
        #[arg(long)]
        build_remote: bool,
    },
    /// Uninstall rbee-hive from local or remote host
    Uninstall {
        /// Host to uninstall from (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        host: String,
        /// Installation directory
        #[arg(short = 'd', long = "dir")]
        install_dir: Option<String>,
    },
    /// Start rbee-hive on local or remote host
    Start {
        /// Host to start on (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        host: String,
        /// Installation directory
        #[arg(short = 'd', long = "dir")]
        install_dir: Option<String>,
        /// HTTP port (default: 7835)
        #[arg(short = 'p', long = "port", default_value = "7835")]
        port: u16,
    },
    /// Stop rbee-hive on local or remote host
    Stop {
        /// Host to stop on (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        host: String,
    },
    /// List all hives
    List,
    /// Get hive details
    Get {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Check hive status
    /// TEAM-189: New command to check if hive is running via health endpoint
    Status {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Refresh device capabilities for a hive
    /// TEAM-196: Fetch and cache device capabilities from a running hive
    /// TEAM-290: Localhost-only (no config files)
    RefreshCapabilities {
        /// Hive alias (only "localhost" supported)
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    /// Run hive-check (narration test through hive SSE)
    /// TEAM-313: Tests narration through hive SSE streaming pipeline
    Check {
        /// Hive alias (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Rebuild rbee-hive from source (update operation)
    /// TEAM-314: Added for parity with queen rebuild
    Rebuild {
        /// Host to rebuild on (default: localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        host: String,
        /// Build on remote host instead of locally (requires git, slower)
        #[arg(long)]
        build_remote: bool,
    },
}
