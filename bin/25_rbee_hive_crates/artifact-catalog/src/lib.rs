// TEAM-273: Shared artifact catalog abstraction
#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-artifact-catalog
//!
//! Shared abstractions for catalog and provisioning patterns.
//! Used by both model-catalog and worker-catalog.

/// Catalog implementations
pub mod catalog;
/// Provisioner abstractions
pub mod provisioner;
/// Core types and traits
pub mod types;

// Re-export main types
pub use catalog::{ArtifactCatalog, FilesystemCatalog};
pub use provisioner::{ArtifactProvisioner, VendorSource};
pub use types::{Artifact, ArtifactStatus};
