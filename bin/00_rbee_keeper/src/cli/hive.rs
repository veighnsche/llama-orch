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
    // TEAM-278: DELETED SshTest, Install, Uninstall
    // These are replaced by package commands (sync, install, uninstall)
    /// List all hives
    List,
    /// Start a hive
    Start {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
    /// Stop a hive
    Stop {
        /// Hive alias (defaults to localhost)
        #[arg(short = 'a', long = "host", default_value = "localhost")]
        alias: String,
    },
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
    RefreshCapabilities {
        /// Hive alias from ~/.config/rbee/hives.conf
        #[arg(short = 'a', long = "host")]
        alias: String,
    },
    // TEAM-278: DELETED ImportSsh - not needed in declarative arch
}
