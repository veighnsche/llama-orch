// TEAM-353: Type conversions between JavaScript and Rust
// Pattern: Same as hive-sdk

use wasm_bindgen::prelude::*;
use operations_contract::Operation;
use serde_wasm_bindgen::{from_value, to_value};

/// Convert JavaScript operation object to Rust Operation
///
/// TEAM-353: Uses serde-wasm-bindgen for automatic conversion
pub fn js_to_operation(js_value: JsValue) -> Result<Operation, JsValue> {
    from_value(js_value)
        .map_err(|e| JsValue::from_str(&format!("Invalid operation: {}", e)))
}

/// Convert Rust error to JavaScript error
///
/// TEAM-353: Simple string conversion
pub fn error_to_js(err: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&err.to_string())
}

/// Convert Rust value to JavaScript value
///
/// TEAM-353: Generic conversion using serde
#[allow(dead_code)]
pub fn to_js<T: serde::Serialize>(value: &T) -> Result<JsValue, JsValue> {
    to_value(value)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}
