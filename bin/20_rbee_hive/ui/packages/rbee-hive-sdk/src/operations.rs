// TEAM-353: Operation builder for Hive operations
// Pattern: Same as queen-rbee-sdk (TEAM-286)
//
// CRITICAL: hive_id is the network address of the Hive (NOT localhost!)
// Queen is localhost, Hives are on the network (192.168.x.x, etc.)

use wasm_bindgen::prelude::*;
use operations_contract::{Operation, WorkerSpawnRequest, WorkerProcessListRequest, WorkerProcessDeleteRequest, ModelListRequest, ModelDownloadRequest, ModelDeleteRequest};
use serde_wasm_bindgen::to_value;

/// Builder for creating Operation objects
///
/// TEAM-353: Provides JavaScript-friendly API for creating operations
/// All methods require hive_id parameter (network address of the Hive)
#[wasm_bindgen]
pub struct OperationBuilder;

#[wasm_bindgen]
impl OperationBuilder {
    /// Create WorkerProcessList operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.workerList('192.168.1.100');
    /// ```
    #[wasm_bindgen(js_name = workerList)]
    pub fn worker_list(hive_id: String) -> JsValue {
        let op = Operation::WorkerProcessList(WorkerProcessListRequest {
            hive_id, // TEAM-353: Network address of the Hive
        });
        to_value(&op).unwrap()
    }

    /// Create WorkerSpawn operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.workerSpawn('192.168.1.100', 'llama-3.2-1b', 'cpu', 0);
    /// ```
    #[wasm_bindgen(js_name = workerSpawn)]
    pub fn worker_spawn(hive_id: String, model: String, worker: String, device: u32) -> JsValue {
        let op = Operation::WorkerSpawn(WorkerSpawnRequest {
            hive_id, // TEAM-353: Network address of the Hive
            model,
            worker,
            device,
        });
        to_value(&op).unwrap()
    }

    /// Create WorkerProcessDelete operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.workerDelete('192.168.1.100', 12345);
    /// ```
    #[wasm_bindgen(js_name = workerDelete)]
    pub fn worker_delete(hive_id: String, pid: u32) -> JsValue {
        let op = Operation::WorkerProcessDelete(WorkerProcessDeleteRequest {
            hive_id, // TEAM-353: Network address of the Hive
            pid,
        });
        to_value(&op).unwrap()
    }

    /// Create ModelList operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.modelList('192.168.1.100');
    /// ```
    #[wasm_bindgen(js_name = modelList)]
    pub fn model_list(hive_id: String) -> JsValue {
        let op = Operation::ModelList(ModelListRequest {
            hive_id, // TEAM-353: Network address of the Hive
        });
        to_value(&op).unwrap()
    }

    /// Create ModelDownload operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.modelDownload('192.168.1.100', 'llama-3.2-1b');
    /// ```
    #[wasm_bindgen(js_name = modelDownload)]
    pub fn model_download(hive_id: String, model: String) -> JsValue {
        let op = Operation::ModelDownload(ModelDownloadRequest {
            hive_id, // TEAM-353: Network address of the Hive
            model,
        });
        to_value(&op).unwrap()
    }

    /// Create ModelDelete operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.modelDelete('192.168.1.100', 'model-id-123');
    /// ```
    #[wasm_bindgen(js_name = modelDelete)]
    pub fn model_delete(hive_id: String, id: String) -> JsValue {
        let op = Operation::ModelDelete(ModelDeleteRequest {
            hive_id, // TEAM-353: Network address of the Hive
            id,
        });
        to_value(&op).unwrap()
    }
}
