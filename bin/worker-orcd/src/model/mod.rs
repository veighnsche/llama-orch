//! Model configuration types
//!
//! This module contains model configuration structures extracted from GGUF metadata.

pub mod gpt_config;
pub mod llama_config;

pub use gpt_config::GPTConfig;
pub use llama_config::LlamaConfig;

// ---
// Implemented by Llama-Beta 🦙
