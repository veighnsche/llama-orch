//! daemon-contract
//!
//! TEAM-315: Generic daemon lifecycle contracts
//!
//! # Purpose
//!
//! This crate provides the foundation types for daemon lifecycle management
//! across the rbee ecosystem. All daemons (queen, hive, workers) use these
//! contracts for consistent lifecycle management.
//!
//! # Components
//!
//! - **DaemonHandle** - Generic handle for all daemons
//! - **Status Types** - Status check protocol
//! - **Install Types** - Installation protocol
//! - **Lifecycle Types** - HTTP daemon configuration
//! - **Shutdown Types** - Shutdown configuration

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Generic daemon handle
pub mod handle;

/// Status types
pub mod status;

/// Installation types
pub mod install;

/// Lifecycle configuration
pub mod lifecycle;

/// Shutdown configuration
pub mod shutdown;

// Re-export main types
pub use handle::DaemonHandle;
pub use install::{InstallConfig, InstallResult, UninstallConfig};
pub use lifecycle::HttpDaemonConfig;
pub use shutdown::ShutdownConfig;
pub use status::{StatusRequest, StatusResponse};
