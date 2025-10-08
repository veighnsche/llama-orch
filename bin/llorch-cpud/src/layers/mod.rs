//! Neural Network Layers
//!
//! IMPORTS: ndarray only (NO worker-crates)
//! Pure implementation - no external dependencies

pub mod attention;
pub mod embedding;
pub mod ffn;
pub mod layer_norm;
pub mod transformer;

pub use attention::Attention;
pub use embedding::Embedding;
pub use ffn::FFN;
pub use layer_norm::LayerNorm;
pub use transformer::TransformerBlock;
