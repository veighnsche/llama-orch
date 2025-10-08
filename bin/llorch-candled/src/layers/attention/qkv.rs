//! QKV Projection for Llama-2
//!
//! Llama-2 uses SEPARATE Q, K, V projections (not combined like GPT-2)
//! - wq: [4096, 4096] - Query projection, no bias
//! - wk: [4096, 4096] - Key projection, no bias
//! - wv: [4096, 4096] - Value projection, no bias
//!
//! Checkpoint 2 validation target
//!
//! Created by: TEAM-000

use ndarray::Array2;

/// Separate Q, K, V projections for Llama-2
///
/// Unlike GPT-2 which uses combined QKV projection,
/// Llama-2 has three separate weight matrices.
pub struct QKVProjection {
    // Stub: will implement in Checkpoint 2
}

impl QKVProjection {
    pub fn new() -> Self {
        todo!("Implement in Checkpoint 2")
    }

    pub fn forward(&self, _x: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        todo!("Implement in Checkpoint 2")
    }
}
