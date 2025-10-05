//! Model adapters for llama-orch workers
//!
//! Provides architecture-specific model adapters that handle different
//! model families (GPT, Llama, Phi-3, Qwen).
//!
//! # Architecture Detection
//!
//! Model architecture is automatically detected from GGUF metadata:
//! - `general.architecture = "llama"` → Llama-style (RoPE, GQA, RMSNorm)
//! - `general.architecture = "gpt2"` → GPT-style (absolute pos, MHA, LayerNorm)
//!
//! # Example
//!
//! ```no_run
//! use worker_models::{ModelAdapter, ModelFactory};
//!
//! let adapter = ModelFactory::from_gguf("model.gguf")?;
//! let config = adapter.config();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// TODO: Extract from worker-orcd/src/models/
// - adapter.rs
// - factory.rs
// - gpt.rs
// - phi3.rs
// - qwen.rs

pub mod placeholder {
    //! Placeholder module until extraction is complete
    
    pub fn version() -> &'static str {
        "0.1.0"
    }
}
