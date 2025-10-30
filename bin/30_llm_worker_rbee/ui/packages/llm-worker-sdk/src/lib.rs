// TEAM-353: Worker WASM SDK - thin wrapper around job-client
// Pattern: Same as hive-sdk and queen-sdk

#![warn(missing_docs)]

//! llm-worker SDK - Rust SDK that compiles to WASM
//!
//! This crate provides JavaScript/TypeScript bindings to the worker system
//! by wrapping existing Rust crates (job-client, job-server, operations-contract)
//! and compiling to WASM.
//!
//! # Architecture
//!
//! ```text
//! job-client + job-server (existing) → llm-worker-sdk (thin wrapper) → WASM → JavaScript
//! ```
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { WorkerClient } from '@rbee/llm-worker-sdk';
//!
//! await init();
//! const client = new WorkerClient('http://localhost:7840', 'worker-1');
//! ```

use wasm_bindgen::prelude::*;

// TEAM-353: Modules (same structure as hive-sdk)
mod client;
mod conversions;

// TEAM-353: Re-export main client
pub use client::WorkerClient;

/// Initialize the WASM module
///
/// TEAM-353: This is called automatically when the WASM module is loaded
#[wasm_bindgen(start)]
pub fn init() {
    // TEAM-353: Panic hook can be added later if needed
}
