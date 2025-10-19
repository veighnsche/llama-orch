//! rbee-hive-device-detection — Runtime GPU detection and information
//!
//! NOTE: Migrated from shared-crates/gpu-info (2025-10-19)
//!
//! Detects available NVIDIA GPUs at runtime, queries VRAM capacity, and enforces GPU-only policy.
//!
//! # Example
//!
//! ```rust,no_run
//! use rbee_hive_device_detection::detect_gpus_or_fail;
//!
//! // Detect GPUs (fail fast if none found)
//! let gpu_info = detect_gpus_or_fail()?;
//!
//! println!("Detected {} GPU(s)", gpu_info.count);
//! for gpu in &gpu_info.devices {
//!     println!("  GPU {}: {} ({} GB VRAM)",
//!         gpu.index, gpu.name, gpu.vram_total_gb());
//! }
//! # Ok::<(), rbee_hive_device_detection::GpuError>(())
//! ```

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::missing_errors_doc)]

mod backend;
mod detection;
mod error;
mod types;

pub use backend::{detect_backends, Backend, BackendCapabilities};
pub use detection::{detect_gpus, detect_gpus_or_fail};
pub use error::{GpuError, Result};
pub use types::{GpuDevice, GpuInfo};

// Convenience functions
pub use detection::{assert_gpu_available, get_device_info, gpu_count, has_gpu};
