// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Module exports

//! rbee-hive library
//!
//! Exposes modules for testing
//!
//! Created by: TEAM-032
//! Modified by: TEAM-034
//! Modified by: TEAM-079 (added worker_provisioner)
//! Modified by: TEAM-104 (added metrics)

pub mod download_tracker;
pub mod metrics; // TEAM-104: Prometheus metrics
pub mod provisioner;
pub mod registry;
pub mod resources; // TEAM-115: Resource monitoring and limits
pub mod worker_provisioner;

// TEAM-079: Re-export model-catalog for convenience
pub use model_catalog;
