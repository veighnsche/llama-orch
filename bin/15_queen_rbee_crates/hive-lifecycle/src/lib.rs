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
//! - `start` - Hive startup (TEAM-212)
//! - `stop` - Hive shutdown (TEAM-212)
//! - `ensure` - Ensure hive running (TEAM-276)
//! - `list` - List all hives (TEAM-211)
//! - `get` - Get hive details (TEAM-211)
//! - `status` - Check hive status (TEAM-211)
//! - `capabilities` - Refresh hive capabilities (TEAM-214)

// TEAM-210: Module declarations
// TEAM-276: Added ensure module
/// Hive capabilities refresh operations
pub mod capabilities;
/// Ensure hive is running (auto-start if needed)
pub mod ensure;
/// Get hive details operations
pub mod get;
/// List all hives operations
pub mod list;
/// SSH utilities for remote operations
pub mod ssh_helper;
// TEAM-278: DELETED pub mod ssh_test, install, uninstall
/// Hive startup operations
pub mod start;
/// Hive status check operations
pub mod status;
/// Hive shutdown operations
pub mod stop;
/// Request/Response types for all operations
pub mod types;
/// Validation helpers
pub mod validation;

// TEAM-212: HTTP client for hive capabilities
/// HTTP client for hive capabilities discovery
pub mod hive_client;

// TEAM-210: Re-export types for convenience
pub use types::*;

// TEAM-278: DELETED ssh_test exports
// TEAM-210: Re-export validation helpers
pub use validation::validate_hive_exists;

// TEAM-211: Export simple operations
pub use get::execute_hive_get;
pub use list::execute_hive_list;
pub use status::execute_hive_status;

// TEAM-212: Export lifecycle operations
pub use start::execute_hive_start;
pub use stop::execute_hive_stop;

// TEAM-278: DELETED install/uninstall exports

// TEAM-214: Export capabilities operation
pub use capabilities::execute_hive_refresh_capabilities;

// TEAM-276: Export ensure operation and HiveHandle
pub use ensure::ensure_hive_running;
