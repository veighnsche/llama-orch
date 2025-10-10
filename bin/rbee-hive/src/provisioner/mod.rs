//! Model provisioner - downloads models from HuggingFace
//!
//! Per test-001-mvp.md Phase 3: Model Provisioning
//! Downloads models using llorch-models script
//!
//! TEAM-030: Simplified to filesystem-based cache (no SQLite)
//! Source of truth is the filesystem - just scans for .gguf files
//!
//! Created by: TEAM-029
//! Modified by: TEAM-030
//! Refactored by: TEAM-033

mod types;
mod catalog;
mod download;
mod operations;

// Re-export public types
pub use types::{DownloadProgress, ModelInfo, ModelProvisioner};
