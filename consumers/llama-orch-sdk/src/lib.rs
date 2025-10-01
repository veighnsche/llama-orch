//! Single-source SDK for llama-orch: Rust core with optional WASM build for npm.
//! This library exposes typed models and a minimal client surface.
//!
//! # Security Notice
//!
//! ⚠️ **API Token Handling**: If you need to handle API tokens or credentials in the SDK,
//! use the `secrets-management` crate (server-side only, not available in WASM):
//!
//! ```rust,ignore
//! #[cfg(not(target_arch = "wasm32"))]
//! use secrets_management::Secret;
//!
//! // Load token securely (server-side only)
//! let token = Secret::load_from_file("/path/to/token")?;
//! ```
//!
//! For WASM/browser: Tokens should be provided by the application, never embedded.

pub mod client;
pub mod types;
