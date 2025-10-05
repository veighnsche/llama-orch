//! Model configuration types
//!
//! This module contains model configuration structures extracted from GGUF metadata.

pub mod llama_config;
pub mod gpt_config;

pub use llama_config::LlamaConfig;
pub use gpt_config::GPTConfig;

// ---
// Implemented by Llama-Beta 🦙
