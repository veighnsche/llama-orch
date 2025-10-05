// Model implementations - LT-022 through LT-031
//
// Provides model-specific implementations for Llama-family models.
// Includes Qwen2.5 and Phi-3 support.
//
// Spec: M0-W-1230, M0-W-1220, M0-W-1214

pub mod qwen;
pub mod phi3;
pub mod adapter;

pub use qwen::{QwenConfig, QwenWeights, QwenModel};
pub use phi3::{Phi3Config, Phi3Weights, Phi3Model};
pub use adapter::{LlamaInferenceAdapter, ModelType, AdapterForwardConfig};
