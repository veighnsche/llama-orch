//! Neural network layers for Llama-2
//!
//! Implements all layers needed for Llama-2 inference:
//! - RMSNorm (Checkpoint 1)
//! - RoPE (Checkpoint 1B)
//! - Separate Q, K, V projections (Checkpoint 2)
//! - SwiGLU FFN (Checkpoint 6)
//! - Attention (Checkpoints 4, 5)
//! - TransformerBlock (Checkpoint 7)
//!
//! Created by: TEAM-000
//! Modified by: TEAM-008 (Removed RoPE struct, now uses apply_rope function)

pub mod rms_norm;
pub mod rope;
pub mod embedding;
pub mod attention;
pub mod swiglu;
pub mod transformer;

pub use rms_norm::RMSNorm;
pub use rope::apply_rope;  // TEAM-008: Changed from RoPE struct to function
pub use embedding::Embedding;
pub use attention::{QKVProjection, Attention};
pub use swiglu::SwiGLU;
pub use transformer::TransformerBlock;
