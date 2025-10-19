// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Backend module structure

//! Inference backend implementation
//!
//! Created by: TEAM-000
//! Refactored by: TEAM-015 (split into focused modules)
//! Modified by: TEAM-017 (added multi-model support with enum pattern)
//! Modified by: TEAM-090 (added GGUF tokenizer extraction)
//! Modified by: TEAM-095 (made `gguf_tokenizer` public for testing)

pub mod gguf_tokenizer;
mod inference;
pub mod models;
mod sampling;
mod tokenizer_loader;

pub use inference::CandleInferenceBackend;
