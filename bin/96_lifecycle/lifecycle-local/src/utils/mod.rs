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
//!
//! TEAM-367: Removed serde.rs - now in lifecycle-shared

pub mod binary;
pub mod local; // TEAM-331: Local process execution (bypasses SSH for localhost)

// Re-export main functions
// TEAM-378: RULE ZERO - Export CheckMode enum and single check_binary_exists() function
// TEAM-379: Export get_binary_mode and is_release_binary for build mode detection
pub use binary::{check_binary_exists, get_binary_mode, is_release_binary, CheckMode};
pub use local::{local_copy, local_exec}; // TEAM-331: Local execution helpers

// TEAM-367: Re-export serde utilities from shared crate
pub use lifecycle_shared::utils::{deserialize_systemtime, serialize_systemtime};
