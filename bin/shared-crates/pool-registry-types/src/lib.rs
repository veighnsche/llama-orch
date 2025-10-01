//! Shared types for pool registry used by both orchestratord and pool-managerd
//!
//! These types enable communication between control plane and GPU nodes in CLOUD_PROFILE
//! while also supporting HOME_PROFILE embedded usage.

pub mod health;
pub mod node;
pub mod pool;

pub use health::{HealthState, HealthStatus};
pub use node::{GpuInfo, NodeCapabilities, NodeId, NodeInfo, NodeStatus};
pub use pool::{PoolId, PoolMetadata, PoolSnapshot};

/// Re-export common error type
pub type Result<T> = anyhow::Result<T>;
