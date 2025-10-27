//! keeper-config-contract
//!
//! TEAM-315: Keeper configuration contract
//!
//! # Purpose
//!
//! This crate provides the configuration schema for rbee-keeper.
//! It ensures configuration stability across versions.
//!
//! # Components
//!
//! - **KeeperConfig** - Main configuration type
//! - **ValidationError** - Configuration validation errors

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Configuration types
pub mod config;

/// Validation errors
pub mod validation;

// Re-export main types
pub use config::KeeperConfig;
pub use validation::ValidationError;
