// Model implementations - LT-022 through LT-031
//
// Provides model-specific implementations for Llama-family models.
// Includes Qwen2.5 and Phi-3 support.
//
// Spec: M0-W-1230, M0-W-1220, M0-W-1214

pub mod adapter;
pub mod factory;
pub mod gpt;
pub mod phi3;
pub mod qwen;

pub use adapter::{AdapterForwardConfig, LlamaModelAdapter, ModelType};
pub use factory::{AdapterFactory, Architecture, FactoryError};
pub use gpt::{GPTConfig, GPTModel};
pub use phi3::{Phi3Config, Phi3Model, Phi3Weights};
pub use qwen::{QwenConfig, QwenModel, QwenWeights};
