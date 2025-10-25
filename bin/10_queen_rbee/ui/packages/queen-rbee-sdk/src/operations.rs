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
    // ========================================================================
    // System Operations (1)
    // ========================================================================

    /// Create a Status operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.status();
    /// ```
    #[wasm_bindgen]
    pub fn status() -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::Status).unwrap()
    }

    // ========================================================================
    // Hive Operations (4)
    // ========================================================================

    /// Create a HiveList operation
    #[wasm_bindgen(js_name = hiveList)]
    pub fn hive_list() -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::HiveList).unwrap()
    }

    /// Create a HiveGet operation
    #[wasm_bindgen(js_name = hiveGet)]
    pub fn hive_get(alias: String) -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::HiveGet { alias }).unwrap()
    }

    /// HiveStatus operation
    #[wasm_bindgen(js_name = hiveStatus)]
    pub fn hive_status(alias: String) -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::HiveStatus { alias }).unwrap()
    }

    /// HiveRefreshCapabilities operation
    #[wasm_bindgen(js_name = hiveRefreshCapabilities)]
    pub fn hive_refresh_capabilities(alias: String) -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::HiveRefreshCapabilities { alias }).unwrap()
    }

    // ========================================================================
    // Worker Process Operations (4)
    // ========================================================================

    /// WorkerSpawn operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.workerSpawn({
    ///   hive_id: 'localhost',
    ///   model: 'llama-3-8b',
    ///   worker: 'cpu',
    ///   device: 0,
    /// });
    /// ```
    #[wasm_bindgen(js_name = workerSpawn)]
    pub fn worker_spawn(params: JsValue) -> Result<JsValue, JsValue> {
        let req: WorkerSpawnRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid worker spawn params: {}", e)))?;
        
        let op = Operation::WorkerSpawn(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    /// WorkerProcessList operation
    #[wasm_bindgen(js_name = workerProcessList)]
    pub fn worker_process_list(params: JsValue) -> Result<JsValue, JsValue> {
        let req: WorkerProcessListRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        
        let op = Operation::WorkerProcessList(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    /// WorkerProcessGet operation
    #[wasm_bindgen(js_name = workerProcessGet)]
    pub fn worker_process_get(params: JsValue) -> Result<JsValue, JsValue> {
        let req: WorkerProcessGetRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        
        let op = Operation::WorkerProcessGet(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    /// WorkerProcessDelete operation
    #[wasm_bindgen(js_name = workerProcessDelete)]
    pub fn worker_process_delete(params: JsValue) -> Result<JsValue, JsValue> {
        let req: WorkerProcessDeleteRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        
        let op = Operation::WorkerProcessDelete(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    // ========================================================================
    // Active Worker Operations (3)
    // ========================================================================

    /// ActiveWorkerList operation
    #[wasm_bindgen(js_name = activeWorkerList)]
    pub fn active_worker_list() -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::ActiveWorkerList).unwrap()
    }

    /// ActiveWorkerGet operation
    #[wasm_bindgen(js_name = activeWorkerGet)]
    pub fn active_worker_get(worker_id: String) -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::ActiveWorkerGet { worker_id }).unwrap()
    }

    /// ActiveWorkerRetire operation
    #[wasm_bindgen(js_name = activeWorkerRetire)]
    pub fn active_worker_retire(worker_id: String) -> JsValue {
        serde_wasm_bindgen::to_value(&Operation::ActiveWorkerRetire { worker_id }).unwrap()
    }

    // ========================================================================
    // Model Operations (4)
    // ========================================================================

    /// ModelDownload operation
    #[wasm_bindgen(js_name = modelDownload)]
    pub fn model_download(params: JsValue) -> Result<JsValue, JsValue> {
        let req: ModelDownloadRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        
        let op = Operation::ModelDownload(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    /// ModelList operation
    #[wasm_bindgen(js_name = modelList)]
    pub fn model_list(params: JsValue) -> Result<JsValue, JsValue> {
        let req: ModelListRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        
        let op = Operation::ModelList(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    /// ModelGet operation
    #[wasm_bindgen(js_name = modelGet)]
    pub fn model_get(params: JsValue) -> Result<JsValue, JsValue> {
        let req: ModelGetRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        
        let op = Operation::ModelGet(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    /// ModelDelete operation
    #[wasm_bindgen(js_name = modelDelete)]
    pub fn model_delete(params: JsValue) -> Result<JsValue, JsValue> {
        let req: ModelDeleteRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid params: {}", e)))?;
        
        let op = Operation::ModelDelete(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    // ========================================================================
    // Inference Operations (1)
    // ========================================================================

    /// Infer operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.infer({
    ///   hive_id: 'localhost',
    ///   model: 'llama-3-8b',
    ///   prompt: 'Hello!',
    ///   max_tokens: 100,
    ///   temperature: 0.7,
    ///   stream: true,
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn infer(params: JsValue) -> Result<JsValue, JsValue> {
        let req: InferRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid infer params: {}", e)))?;

        let op = Operation::Infer(req);
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }
}
