//! Attention Scores Computation
//!
//! Computes scaled dot-product attention scores:
//! scores = (Q @ K.T) / sqrt(head_dim)
//!
//! For Llama-2:
//! - head_dim = 128 (not 64 like GPT-2)
//! - scale = 1/sqrt(128) = 0.0884 (not 0.125)
//!
//! Checkpoint 4 validation target
//!
//! Created by: TEAM-000

use ndarray::Array3;

/// Attention scores computation
pub struct AttentionScores {
    // Stub: will implement in Checkpoint 4
}

impl AttentionScores {
    pub fn new() -> Self {
        todo!("Implement in Checkpoint 4")
    }

    pub fn forward(&self, _q: &Array3<f32>, _k: &Array3<f32>) -> Array3<f32> {
        todo!("Implement in Checkpoint 4")
    }
}
