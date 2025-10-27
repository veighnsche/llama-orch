//! Utility modules
//!
//! TEAM-329: Utilities that don't fit operation pattern
//! Config types are in types/, not here

pub mod find;
pub mod paths; // TEAM-329: Moved from src/paths.rs
pub mod pid; // TEAM-329: Centralized PID operations
pub mod poll; // TEAM-329: Extracted from health.rs
pub mod serde; // TEAM-329: Serde helpers (extracted from types/install.rs)
pub mod timeout;

// Re-export main functions
pub use find::find_binary;
pub use paths::{get_install_dir, get_install_path};
pub use pid::{get_pid_file_path, read_pid_file, remove_pid_file, write_pid_file};
pub use poll::poll_daemon_health;
pub use timeout::{timeout_after, with_timeout};
