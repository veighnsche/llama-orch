// RHAI Script Management
// Provides client-side API for RHAI script operations via job-based architecture
//
// All operations submit jobs to /v1/jobs endpoint

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use job_client::JobClient;
use operations_contract::Operation;
use crate::conversions::error_to_js;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct RhaiScript {
    id: Option<String>,
    name: String,
    content: String,
    created_at: Option<String>,
    updated_at: Option<String>,
}

#[wasm_bindgen]
impl RhaiScript {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String, content: String) -> Self {
        Self {
            id: None,
            name,
            content,
            created_at: None,
            updated_at: None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn id(&self) -> Option<String> {
        self.id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn content(&self) -> String {
        self.content.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn created_at(&self) -> Option<String> {
        self.created_at.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn updated_at(&self) -> Option<String> {
        self.updated_at.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct TestResult {
    success: bool,
    output: Option<String>,
    error: Option<String>,
}

#[wasm_bindgen]
impl TestResult {
    #[wasm_bindgen(getter)]
    pub fn success(&self) -> bool {
        self.success
    }

    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Option<String> {
        self.output.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn error(&self) -> Option<String> {
        self.error.clone()
    }
}

/// RHAI Script Client
/// 
/// Manages RHAI scripts via Queen API using job-client
#[wasm_bindgen]
pub struct RhaiClient {
    base_url: String,
}

impl RhaiClient {
    /// Submit an operation and return the result
    async fn submit_operation(&self, operation: Operation) -> Result<serde_json::Value, String> {
        let client = job_client::JobClient::new(&self.base_url);
        
        // Submit the job and get the job_id
        let job_id = client.submit(operation)
            .await
            .map_err(|e| format!("Failed to submit job: {}", e))?;
        
        // For now, just return success with job_id
        // TODO: Wait for job completion and return actual result
        Ok(serde_json::json!({
            "job_id": job_id,
            "status": "submitted"
        }))
    }
}

#[wasm_bindgen]
impl RhaiClient {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self { base_url }
    }

    /// Save a RHAI script via job submission
    #[wasm_bindgen(js_name = saveScript)]
    pub async fn save_script(&self, script: JsValue) -> Result<JsValue, JsValue> {
        let script: RhaiScript = serde_wasm_bindgen::from_value(script)?;
        
        let operation = Operation::RhaiScriptSave {
            name: script.name,
            content: script.content,
            id: script.id,
        };
        
        let result = self.submit_operation(operation)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Test a RHAI script via job submission
    #[wasm_bindgen(js_name = testScript)]
    pub async fn test_script(&self, content: String) -> Result<JsValue, JsValue> {
        let operation = Operation::RhaiScriptTest { content };
        
        let result = self.submit_operation(operation)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Get a RHAI script by ID via job submission
    #[wasm_bindgen(js_name = getScript)]
    pub async fn get_script(&self, id: String) -> Result<JsValue, JsValue> {
        let operation = Operation::RhaiScriptGet { id };
        
        let result = self.submit_operation(operation)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// List all RHAI scripts via job submission
    #[wasm_bindgen(js_name = listScripts)]
    pub async fn list_scripts(&self) -> Result<JsValue, JsValue> {
        let operation = Operation::RhaiScriptList;
        
        let result = self.submit_operation(operation)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        
        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Delete a RHAI script via job submission
    #[wasm_bindgen(js_name = deleteScript)]
    pub async fn delete_script(&self, id: String) -> Result<(), JsValue> {
        let operation = Operation::RhaiScriptDelete { id };
        
        self.submit_operation(operation)
            .await
            .map_err(|e| JsValue::from_str(&e))?;
        
        Ok(())
    }
}
