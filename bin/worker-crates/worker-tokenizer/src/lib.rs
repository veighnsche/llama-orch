//! Tokenization for llama-orch workers
//!
//! Provides two tokenizer backends:
//! - GGUF byte-BPE: Embedded in GGUF files (Qwen, Phi-3, Llama)
//! - HuggingFace JSON: External tokenizer.json (GPT-OSS-20B)
//!
//! # Example
//!
//! ```no_run
//! use worker_tokenizer::{Tokenizer, TokenizerBackend};
//!
//! let tokenizer = Tokenizer::from_gguf("model.gguf")?;
//! let tokens = tokenizer.encode("Hello world")?;
//! let text = tokenizer.decode(&tokens)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// TODO: Extract from worker-orcd/src/tokenizer/
// - backend.rs
// - decoder.rs
// - encoder.rs
// - hf_json.rs
// - streaming.rs
// - vocab.rs
// - merges.rs
// - metadata.rs
// - error.rs
// - discovery.rs

pub mod placeholder {
    //! Placeholder module until extraction is complete
    
    pub fn version() -> &'static str {
        "0.1.0"
    }
}
