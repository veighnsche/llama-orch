# TEAM-286: Phase 3 - All Operations

**Phase:** 3 of 4  
**Duration:** 1-2 days  
**Status:** 📋 **READY TO START**

---

## Goal

Expose all remaining operations from `operations-contract` as JavaScript-friendly builders and add convenience methods to RbeeClient.

---

## Prerequisites

- ✅ Phase 2 completed (Core bindings)
- ✅ submit_and_stream() working
- ✅ OperationBuilder pattern established

---

## Deliverables

1. **Complete OperationBuilder**
   - All 17 operations
   - Type-safe builders
   - JSDoc comments

2. **Convenience Methods**
   - High-level API on RbeeClient
   - Common use cases

3. **Comprehensive Examples**
   - One example per operation category
   - Real-world usage

---

## Operations to Implement

From `operations-contract`:

**System (1):**
- ✅ Status (done in Phase 2)

**Hive (4):**
- ✅ HiveList (done)
- ✅ HiveGet (done)
- HiveStatus
- HiveRefreshCapabilities

**Worker Process (4):**
- WorkerSpawn
- WorkerProcessList
- WorkerProcessGet
- WorkerProcessDelete

**Active Worker (3):**
- ActiveWorkerList
- ActiveWorkerGet
- ActiveWorkerRetire

**Model (4):**
- ModelDownload
- ModelList
- ModelGet
- ModelDelete

**Inference (1):**
- ✅ Infer (done in Phase 2)

---

## Step-by-Step Implementation

### Step 1: Complete OperationBuilder (3 hours)

**File:** `consumers/rbee-sdk/src/operations.rs` (extend)

```rust
// TEAM-286: Complete operation builders

use wasm_bindgen::prelude::*;
use operations_contract::*;
use serde_wasm_bindgen;

#[wasm_bindgen]
impl OperationBuilder {
    // ========================================================================
    // Hive Operations
    // ========================================================================

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
    // Worker Process Operations
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
    // Active Worker Operations
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
    // Model Operations
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
}
```

---

### Step 2: Convenience Methods (2 hours)

**File:** `consumers/rbee-sdk/src/client.rs` (extend)

```rust
// TEAM-286: Add convenience methods

#[wasm_bindgen]
impl RbeeClient {
    // ... existing methods ...

    /// Convenience: Get system status
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const status = await client.status();
    /// ```
    #[wasm_bindgen]
    pub async fn status(&self, callback: js_sys::Function) -> Result<String, JsValue> {
        use crate::operations::OperationBuilder;
        let op = OperationBuilder::status();
        self.submit_and_stream(op, callback).await
    }

    /// Convenience: List hives
    #[wasm_bindgen(js_name = listHives)]
    pub async fn list_hives(&self, callback: js_sys::Function) -> Result<String, JsValue> {
        use crate::operations::OperationBuilder;
        let op = OperationBuilder::hive_list();
        self.submit_and_stream(op, callback).await
    }

    /// Convenience: Run inference
    ///
    /// # JavaScript Example
    /// ```javascript
    /// await client.infer({
    ///   model: 'llama-3-8b',
    ///   prompt: 'Hello!',
    ///   max_tokens: 100,
    /// }, (token) => console.log(token));
    /// ```
    #[wasm_bindgen]
    pub async fn infer(
        &self,
        params: JsValue,
        callback: js_sys::Function,
    ) -> Result<String, JsValue> {
        use crate::operations::OperationBuilder;
        let op = OperationBuilder::infer(params)?;
        self.submit_and_stream(op, callback).await
    }
}
```

---

### Step 3: Examples for All Operations (3 hours)

**File:** `consumers/rbee-sdk/examples/all-operations.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>rbee SDK - All Operations</title>
    <style>
        body { font-family: monospace; padding: 20px; }
        button { margin: 5px; padding: 10px; }
        pre { background: #f5f5f5; padding: 10px; }
    </style>
</head>
<body>
    <h1>🐝 rbee SDK - All Operations</h1>
    
    <h2>System</h2>
    <button onclick="testStatus()">Status</button>
    
    <h2>Hive</h2>
    <button onclick="testHiveList()">List Hives</button>
    <button onclick="testHiveGet()">Get Hive</button>
    <button onclick="testHiveStatus()">Hive Status</button>
    
    <h2>Worker</h2>
    <button onclick="testWorkerSpawn()">Spawn Worker</button>
    <button onclick="testWorkerList()">List Workers</button>
    <button onclick="testActiveWorkers()">Active Workers</button>
    
    <h2>Model</h2>
    <button onclick="testModelList()">List Models</button>
    <button onclick="testModelDownload()">Download Model</button>
    
    <h2>Inference</h2>
    <button onclick="testInference()">Run Inference</button>
    
    <div id="output"></div>
    
    <script type="module">
        import init, { RbeeClient, OperationBuilder } from '../pkg/rbee_sdk.js';
        
        await init();
        const client = new RbeeClient('http://localhost:8500');
        window.client = client;
        
        function log(msg) {
            document.getElementById('output').innerHTML = `<pre>${msg}</pre>`;
        }
        
        window.testStatus = async () => {
            log('Getting status...');
            const lines = [];
            await client.status((line) => {
                lines.push(line);
                log(lines.join('\n'));
            });
        };
        
        window.testHiveList = async () => {
            log('Listing hives...');
            const lines = [];
            await client.listHives((line) => {
                lines.push(line);
                log(lines.join('\n'));
            });
        };
        
        window.testWorkerSpawn = async () => {
            log('Spawning worker...');
            const op = OperationBuilder.workerSpawn({
                hive_id: 'localhost',
                model: 'llama-3-8b',
                worker: 'cpu',
                device: 0,
            });
            
            const lines = [];
            await client.submitAndStream(op, (line) => {
                lines.push(line);
                log(lines.join('\n'));
            });
        };
        
        window.testModelList = async () => {
            log('Listing models...');
            const op = OperationBuilder.modelList({ hive_id: 'localhost' });
            
            const lines = [];
            await client.submitAndStream(op, (line) => {
                lines.push(line);
                log(lines.join('\n'));
            });
        };
        
        window.testInference = async () => {
            log('Running inference...');
            await client.infer({
                hive_id: 'localhost',
                model: 'llama-3-8b',
                prompt: 'Write a haiku about Rust',
                max_tokens: 50,
                temperature: 0.7,
                stream: true,
            }, (line) => {
                log(line);
            });
        };
        
        // TODO: Add other operation tests
        
        console.log('✅ All operations loaded');
    </script>
</body>
</html>
```

---

### Step 4: Documentation (1 hour)

**File:** `consumers/rbee-sdk/README.md`

```markdown
# rbee SDK (Rust + WASM)

Rust SDK that compiles to WASM for browser and Node.js usage.

## Installation

```bash
npm install @rbee/sdk
```

## Usage

### Browser

```html
<script type="module">
import init, { RbeeClient, OperationBuilder } from '@rbee/sdk';

await init();
const client = new RbeeClient('http://localhost:8500');

// Run inference
await client.infer({
  model: 'llama-3-8b',
  prompt: 'Hello!',
  max_tokens: 100,
}, (token) => console.log(token));
</script>
```

### Node.js

```javascript
const { RbeeClient } = require('@rbee/sdk');

const client = new RbeeClient('http://localhost:8500');
await client.health();
```

## API

### RbeeClient

- `new RbeeClient(baseUrl)` - Create client
- `health()` - Health check
- `submit(operation)` - Submit job
- `submitAndStream(operation, callback)` - Submit and stream
- `infer(params, callback)` - Convenience for inference
- `status(callback)` - Get system status
- `listHives(callback)` - List hives

### OperationBuilder

- `OperationBuilder.status()` - System status
- `OperationBuilder.infer(params)` - Inference
- `OperationBuilder.workerSpawn(params)` - Spawn worker
- `OperationBuilder.modelDownload(params)` - Download model
- And 12+ more...

## Examples

See `examples/` directory.

## Building

```bash
wasm-pack build --target web
```

## License

GPL-3.0-or-later
```

---

## Verification Checklist

After completing Phase 3:

- [ ] All 17 operations have builders
- [ ] All builders work correctly
- [ ] Convenience methods implemented
- [ ] Examples for all operation categories
- [ ] Documentation complete
- [ ] All operations tested
- [ ] TypeScript types generated
- [ ] All TEAM-286 signatures added

---

## Next Phase

**Phase 4: Publishing** - Build, package, and publish to npm.

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Estimated Duration:** 1-2 days
