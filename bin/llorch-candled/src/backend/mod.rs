//! Inference backend implementation
//!
//! Created by: TEAM-000
//! Refactored by: TEAM-015 (split into focused modules)

mod model_loader;
mod sampling;
mod inference;

pub use inference::CandleInferenceBackend;
