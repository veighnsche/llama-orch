//! Narration module for model-loader
//!
//! Provides structured, human-readable narration for all model loading operations.
//! 
//! Uses `observability-narration-core` to emit developer-friendly debugging stories
//! in both human-readable and cute children's book modes! 🎀

pub mod events;

pub use events::*;
