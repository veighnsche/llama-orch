//! Utility modules
//!
//! TEAM-329: Utilities that don't fit operation pattern
//! Config types are in types/, not here
//! TEAM-330: Added ssh module, removed local-only utilities (find, paths, pid, timeout)
//!
//! TEAM-330: CLEANUP - Removed unused local-only utilities:
//! - find.rs: Local binary finding (not for remote)
//! - paths.rs: Local path helpers (not for remote)
//! - pid.rs: Local PID file management (not for remote)
//! - timeout.rs: Obsolete (replaced by #[with_timeout] attribute)

pub mod poll; // TEAM-329: Extracted from health.rs - HTTP health polling (remote-compatible)
pub mod serde; // TEAM-329: Serde helpers (used by types/install.rs for timestamps)
pub mod ssh; // TEAM-330: SSH/SCP operations (core remote functionality)

// Re-export main functions
pub use poll::{poll_daemon_health, HealthPollConfig}; // TEAM-330: Moved from types/
pub use ssh::{ssh_exec, scp_upload}; // TEAM-330: SSH helpers
