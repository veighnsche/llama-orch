// TEAM-286: Main library entry point for WASM SDK

#![warn(missing_docs)]

//! rbee SDK - Rust SDK that compiles to WASM
//!
//! This crate provides JavaScript/TypeScript bindings to the rbee system
//! by wrapping existing Rust crates (job-client, operations-contract, etc.)
//! and compiling to WASM.
//!
//! # Architecture
//!
//! ```text
//! job-client (existing) → rbee-sdk (thin wrapper) → WASM → JavaScript
//! ```
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { QueenClient } from '@rbee/queen-rbee-sdk';
//!
//! await init();
//! const client = new QueenClient('http://localhost:7833');
//! await client.submitAndStream(operation, (line) => console.log(line));
//! ```

use wasm_bindgen::prelude::*;

// TEAM-286: Modules
mod client;
mod conversions;
mod heartbeat;
mod operations;
mod rhai;

// TEAM-286: Re-export main client, operation builder, and heartbeat monitor
pub use client::QueenClient;
pub use heartbeat::HeartbeatMonitor;
pub use operations::OperationBuilder;
pub use rhai::{RhaiClient, RhaiScript, TestResult};

/// Initialize the WASM module
///
/// TEAM-286: This is called automatically when the WASM module is loaded
#[wasm_bindgen(start)]
pub fn init() {
    // TEAM-286: Panic hook will be added in Phase 2 if needed
}
