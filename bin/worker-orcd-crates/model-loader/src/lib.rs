//! model-loader — Model validation and loading
//!
//! Validates model files (signature, hash, format) before loading into VRAM.
//!
//! # Security Properties
//!
//! - Cryptographic signature verification
//! - Hash validation (SHA-256)
//! - GGUF format validation
//! - Size limit enforcement
//! - Fail-fast on invalid models
//!
//! # ⚠️ INPUT VALIDATION REMINDER
//!
//! **Validate model paths and hashes** with `input-validation`:
//!
//! ```rust,ignore
//! use input_validation::{validate_path, validate_hex_string, validate_identifier};
//!
//! // Validate model file path
//! validate_path(&model_path, &allowed_models_dir)?;
//!
//! // Validate SHA-256 hash
//! validate_hex_string(&hash, 64)?;
//!
//! // Validate model ID
//! validate_identifier(&model_id, 256)?;
//! ```
//!
//! See: `bin/shared-crates/input-validation/README.md`
//!
//! # Example
//!
//! ```rust,ignore
//! use model_loader::{ModelLoader, LoadRequest};
//! use std::path::Path;
//!
//! let loader = ModelLoader::new();
//!
//! let request = LoadRequest::new(Path::new("/models/llama-3.1-8b.gguf"))
//!     .with_hash("abc123...")
//!     .with_max_size(10_000_000_000); // 10GB
//!
//! let model_bytes = loader.load_and_validate(request)?;
//! ```
//!
//! # Module Organization
//!
//! - `error` — Error types with actionable diagnostics
//! - `types` — Core types (LoadRequest, etc.)
//! - `loader` — Main ModelLoader implementation
//! - `validation` — Validation modules:
//!   - `hash` — SHA-256 hash verification
//!   - `path` — Filesystem path validation
//!   - `gguf` — GGUF format validation
//!     - `parser` — Bounds-checked GGUF parsing
//!     - `limits` — Security limits (MAX_TENSORS, etc.)

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]
#![allow(dead_code)] // Allow during development

// Public modules
pub mod error;
pub mod types;
pub mod loader;
pub mod validation;

// Observability
pub mod narration;

// Re-exports for convenience
pub use error::{LoadError, Result};
pub use types::LoadRequest;
pub use loader::ModelLoader;
