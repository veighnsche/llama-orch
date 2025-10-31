//! Model Provisioner - Downloads models from HuggingFace
//!
//! Uses the official `hf-hub` Rust crate (same library used by Candle)
//! to download GGUF models from HuggingFace Hub.
//!
//! # Architecture
//!
//! ```text
//! ModelProvisioner
//!     ↓
//! HuggingFaceVendor (implements VendorSource)
//!     ↓
//! hf-hub crate (official HuggingFace Rust client)
//!     ↓
//! Downloads to ~/.cache/rbee/models/{model_id}/
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod huggingface;
mod provisioner;

pub use huggingface::HuggingFaceVendor;
pub use provisioner::ModelProvisioner;
