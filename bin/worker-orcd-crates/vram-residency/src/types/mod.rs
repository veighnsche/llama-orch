//! Type definitions for VRAM residency
//!
//! This module contains all core types used throughout the crate.

pub mod sealed_shard;
pub mod vram_config;

pub use sealed_shard::SealedShard;
pub use vram_config::VramConfig;
