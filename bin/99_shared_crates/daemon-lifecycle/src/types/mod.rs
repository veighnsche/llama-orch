//! Daemon lifecycle types
//!
//! TEAM-329: Inlined from daemon-contract (RULE ZERO - 1 consumer = inline it)
//! TEAM-329: PERFECT PARITY - types/{operation}.rs matches src/{operation}.rs
//!
//! These types were previously in a separate "daemon-contract" crate, but since
//! daemon-lifecycle was the ONLY consumer, they've been inlined to eliminate
//! unnecessary indirection.
//!
//! **PARITY RULE:** Every operation in src/ has a corresponding types/ file

pub mod build; // (no types - simple operation)
pub mod install; // InstallConfig, InstallResult
pub mod rebuild; // RebuildConfig
pub mod shutdown; // ShutdownConfig
pub mod start; // HttpDaemonConfig
pub mod status; // StatusRequest, StatusResponse, HealthPollConfig
pub mod stop; // (no unique types - uses HttpDaemonConfig from start)
pub mod timeout; // TimeoutConfig
pub mod uninstall; // UninstallConfig

// Re-export main types
pub use install::{InstallConfig, InstallResult};
pub use rebuild::RebuildConfig;
pub use shutdown::ShutdownConfig;
pub use start::HttpDaemonConfig;
pub use status::{HealthPollConfig, StatusRequest, StatusResponse};
pub use timeout::TimeoutConfig;
pub use uninstall::UninstallConfig;
