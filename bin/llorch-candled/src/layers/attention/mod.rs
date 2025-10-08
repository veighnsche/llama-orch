//! Attention module for Llama-2
//!
//! Split into focused files for clarity:
//! - qkv.rs: Separate Q, K, V projections (Checkpoint 2)
//! - scores.rs: Attention score computation (Checkpoint 4)
//! - output.rs: Attention output projection (Checkpoint 5)
//!
//! Note: KVCache is in src/cache/ (top-level module for future growth)
//!
//! Created by: TEAM-000

mod qkv;
mod scores;
mod output;

pub use qkv::QKVProjection;
pub use scores::AttentionScores;
pub use output::AttentionOutput;

// Main Attention struct will orchestrate all components
// Will be implemented after individual checkpoints pass
// pub struct Attention { ... }
