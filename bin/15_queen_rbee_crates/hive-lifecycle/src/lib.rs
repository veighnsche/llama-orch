// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-164: Implemented hive lifecycle management
// TEAM-210: Phase 1 Foundation - Module structure and types
// TEAM-220: Investigated - Behavior inventory complete
// Purpose: Lifecycle management for rbee-hive instances

#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-rbee-hive-lifecycle
//!
//! **Category:** Orchestration
//! **Pattern:** Command Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Lifecycle management for rbee-hive instances.
//! Queen orchestrates hive spawning - rbee-keeper does NOT.
//!
//! # Module Structure
//!
//! - `types` - Request/Response types for all operations
//! - `validation` - Validation helpers (validate_hive_exists)
//! - `ssh_test` - SSH connection testing
//! - `install` - Hive installation (TEAM-213)
//! - `uninstall` - Hive uninstallation (TEAM-213)
//! - `start` - Hive startup (TEAM-212)
//! - `stop` - Hive shutdown (TEAM-212)
//! - `list` - List all hives (TEAM-211)
//! - `get` - Get hive details (TEAM-211)
//! - `status` - Check hive status (TEAM-211)
//! - `capabilities` - Refresh hive capabilities (TEAM-214)

// TEAM-210: Module declarations
pub mod capabilities;
pub mod get;
pub mod install;
pub mod list;
pub mod ssh_test;
pub mod start;
pub mod status;
pub mod stop;
pub mod types;
pub mod uninstall;
pub mod validation;

// TEAM-212: HTTP client for hive capabilities
pub mod hive_client;

// TEAM-210: Re-export types for convenience
pub use types::*;

// TEAM-210: Re-export SSH test interface
pub use ssh_test::{execute_ssh_test, SshTestRequest, SshTestResponse};

// TEAM-210: Re-export validation helpers
pub use validation::validate_hive_exists;

// TEAM-211: Export simple operations
pub use get::execute_hive_get;
pub use list::execute_hive_list;
pub use status::execute_hive_status;

// TEAM-212: Export lifecycle operations
pub use start::execute_hive_start;
pub use stop::execute_hive_stop;

// TEAM-213: Export install/uninstall operations
pub use install::execute_hive_install;
pub use uninstall::execute_hive_uninstall;

// TEAM-214: Export capabilities operation
pub use capabilities::execute_hive_refresh_capabilities;
