//! Pool lifecycle management.
//!
//! Handles spawning, draining, reloading, and supervising engine processes.

pub mod preload;
pub mod drain;
pub mod reload;
pub mod supervision;
