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
//! TEAM-358: RULE ZERO FIX - Removed poll.rs and ssh.rs:
//! - poll.rs: Duplicated code, use health-poll crate instead
//! - ssh.rs: lifecycle-local manages LOCAL daemons only, no SSH needed

pub mod binary;
pub mod local; // TEAM-331: Local process execution (bypasses SSH for localhost)
pub mod serde; // TEAM-329: Serde helpers (used by types/install.rs for timestamps)

// Re-export main functions
pub use binary::check_binary_installed;
pub use local::{local_copy, local_exec}; // TEAM-331: Local execution helpers
