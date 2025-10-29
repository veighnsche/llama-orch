// TEAM-286: Conversion helpers between Rust and JavaScript
//
// Note: This file contains conversion functions, not type definitions.
// The actual types are defined in operations-contract and other crates.

use wasm_bindgen::prelude::*;
use operations_contract::Operation;

/// Convert JavaScript operation to Rust Operation
///
/// TEAM-286: Uses serde-wasm-bindgen for automatic conversion
pub fn js_to_operation(js_value: JsValue) -> Result<Operation, JsValue> {
    serde_wasm_bindgen::from_value(js_value)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse operation: {}", e)))
}

/// Convert Rust error to JavaScript error
pub fn error_to_js(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}
