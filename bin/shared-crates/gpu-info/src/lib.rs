//! gpu-info — Runtime GPU detection and information
//!
//! Detects available NVIDIA GPUs at runtime, queries VRAM capacity, and enforces GPU-only policy.
//!
//! # Example
//!
//! ```rust
//! use gpu_info::detect_gpus_or_fail;
//!
//! // Detect GPUs (fail fast if none found)
//! let gpu_info = detect_gpus_or_fail()?;
//!
//! println!("Detected {} GPU(s)", gpu_info.count);
//! for gpu in &gpu_info.devices {
//!     println!("  GPU {}: {} ({} GB VRAM)",
//!         gpu.index, gpu.name, gpu.vram_total_gb());
//! }
//! # Ok::<(), gpu_info::GpuError>(())
//! ```

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::missing_errors_doc)]

mod detection;
mod error;
mod types;

pub use detection::{detect_gpus, detect_gpus_or_fail};
pub use error::{GpuError, Result};
pub use types::{GpuDevice, GpuInfo};

// Convenience functions
pub use detection::{assert_gpu_available, get_device_info, gpu_count, has_gpu};
