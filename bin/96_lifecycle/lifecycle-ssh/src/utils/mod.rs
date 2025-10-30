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
//!
//! TEAM-358: RULE ZERO FIX - Removed poll.rs:
//! - poll.rs: Duplicated code, use health-poll crate instead

pub mod binary;
pub mod local; // TEAM-358: Local helpers (NOT for localhost bypass - kept for potential future use)
pub mod serde; // TEAM-329: Serde helpers (used by types/install.rs for timestamps)
pub mod ssh; // TEAM-330: SSH/SCP operations (core remote functionality)

// Re-export main functions
pub use binary::check_binary_installed;
// TEAM-358: local_copy and local_exec are NOT exported - use lifecycle-local for local operations
pub use ssh::{scp_upload, ssh_exec}; // TEAM-330: SSH helpers
