// TEAM-286: Operation builder helpers for JavaScript

use wasm_bindgen::prelude::*;
use operations_contract::*;
use serde_wasm_bindgen;

/// Operation builders for JavaScript
///
/// TEAM-286: These provide convenient ways to construct operations
/// from JavaScript without manually building objects
#[wasm_bindgen]
pub struct OperationBuilder;

#[wasm_bindgen]
impl OperationBuilder {
    // ═══════════════════════════════════════════════════════════════════════
    // QUEEN OPERATIONS
    // ═══════════════════════════════════════════════════════════════════════
    // Queen UI only needs:
    // 1. Status - for heartbeat monitoring
    // 2. QueenCheck - for diagnostics
    //
    // All worker/model/infer operations belong to Hive UI
    // ═══════════════════════════════════════════════════════════════════════

    /// Create a Status operation
    ///
    /// Returns live status of all workers from the registry.
    /// Used by the heartbeat monitor UI.
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.status();
    /// ```
    #[wasm_bindgen]
    pub fn status() -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::Status).unwrap()
    }

    /// Create a QueenCheck operation
    ///
    /// Deep narration test through queen job server.
    /// Used for diagnostics.
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.queenCheck();
    /// ```
    #[wasm_bindgen(js_name = queenCheck)]
    pub fn queen_check() -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::QueenCheck).unwrap()
    }
}
