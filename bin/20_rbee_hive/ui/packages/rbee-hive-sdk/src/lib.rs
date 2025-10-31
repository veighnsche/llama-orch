// TEAM-353: Hive WASM SDK - thin wrapper around job-client
// Pattern: Same as queen-rbee-sdk (TEAM-286)

#![warn(missing_docs)]

//! rbee-hive SDK - Rust SDK that compiles to WASM
//!
//! This crate provides JavaScript/TypeScript bindings to the rbee-hive system
//! by wrapping existing Rust crates (job-client, operations-contract, etc.)
//! and compiling to WASM.
//!
//! # Architecture
//!
//! ```text
//! job-client (existing) → rbee-hive-sdk (thin wrapper) → WASM → JavaScript
//! ```
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { HiveClient } from '@rbee/rbee-hive-sdk';
//!
//! await init();
//! const client = new HiveClient('http://localhost:7835');
//! await client.submitAndStream(operation, (line) => console.log(line));
//! ```

use wasm_bindgen::prelude::*;

// TEAM-353: Modules (same structure as Queen SDK)
mod client;
mod conversions;
mod operations;
mod heartbeat; // TEAM-374: Heartbeat monitoring

// TEAM-353: Re-export main client and operation builder
pub use client::HiveClient;
pub use operations::OperationBuilder;
pub use heartbeat::HeartbeatMonitor; // TEAM-374: Heartbeat monitoring

/// Initialize the WASM module
///
/// TEAM-286: This is called automatically when the WASM module is loaded
/// TEAM-377: Added console logging to verify WASM loads
#[wasm_bindgen(start)]
pub fn init() {
    // TEAM-377: Log to console so we know WASM loaded
    web_sys::console::log_1(&"🎉 [Hive SDK] WASM module initialized successfully!".into());
}
