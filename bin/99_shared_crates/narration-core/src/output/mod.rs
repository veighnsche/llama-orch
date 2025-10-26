// TEAM-300: Modular reorganization - Output module
//! Output mechanisms for narration
//!
//! This module contains the various output mechanisms:
//! - SSE (Server-Sent Events) for web UI streaming
//! - Capture adapter for testing

pub mod capture;
pub mod sse_sink;

pub use capture::{CaptureAdapter, CapturedNarration};
pub use sse_sink::NarrationEvent;
