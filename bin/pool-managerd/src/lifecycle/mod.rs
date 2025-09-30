//! Pool lifecycle management.
//!
//! Handles spawning, draining, reloading, and supervising engine processes.

pub mod drain;
pub mod preload;
pub mod reload;
pub mod supervision;
