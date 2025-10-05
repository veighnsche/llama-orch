//! Inference module
//!
//! Architecture-specific inference adapters for different model architectures

pub mod cuda_backend;
pub mod gpt_adapter;
// pub mod qwen_inference;  // TODO: Complete integration
pub use gpt_adapter::{GPTModelAdapter, QuantizationType};

// ---
// Crafted by GPT-Gamma 🤖
