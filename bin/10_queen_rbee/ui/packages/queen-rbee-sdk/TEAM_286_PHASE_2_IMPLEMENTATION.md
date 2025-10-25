# TEAM-286: Phase 2 - Core Bindings

**Phase:** 2 of 4  
**Duration:** 2 days  
**Status:** 📋 **READY TO START**

---

## Goal

Wrap the existing `job-client` crate for WASM, exposing `submit_and_stream()` and other core methods to JavaScript with proper async handling.

---

## Prerequisites

- ✅ Phase 1 completed (WASM setup)
- ✅ WASM compilation working
- ✅ Understanding of wasm-bindgen async

---

## Deliverables

1. **Async Job Submission**
   - Wrap JobClient::submit_and_stream()
   - Handle JavaScript callbacks
   - Proper error propagation

2. **Operation Handling**
   - Convert JS operations to Rust
   - Expose operations-contract types
   - Type-safe API

3. **Examples**
   - Basic job submission
   - Streaming example
   - Error handling

---

## Key Challenge: Async + Callbacks in WASM

**Problem:** Rust async + JavaScript callbacks requires special handling

**Solution:** Use `wasm-bindgen-futures` + `js_sys::Function`

---

## Step-by-Step Implementation

### Step 1: Extend RbeeClient (3 hours)

**File:** `consumers/rbee-sdk/src/client.rs`

```rust
// TEAM-286: Complete RbeeClient implementation

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::spawn_local;
use job_client::JobClient;
use operations_contract::Operation;
use crate::types::{js_to_operation, error_to_js};

#[wasm_bindgen]
pub struct RbeeClient {
    inner: JobClient,
}

#[wasm_bindgen]
impl RbeeClient {
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            inner: JobClient::new(base_url),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn base_url(&self) -> String {
        self.inner.base_url().to_string()
    }

    /// Submit a job and stream results
    ///
    /// TEAM-286: This wraps JobClient::submit_and_stream() from job-client crate
    ///
    /// # Arguments
    /// * `operation` - JavaScript object representing an operation
    /// * `on_line` - JavaScript callback function called for each line
    ///
    /// # Returns
    /// Promise that resolves to job_id
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const jobId = await client.submitAndStream(
    ///   { operation: 'status' },
    ///   (line) => console.log(line)
    /// );
    /// ```
    #[wasm_bindgen(js_name = submitAndStream)]
    pub async fn submit_and_stream(
        &self,
        operation: JsValue,
        on_line: js_sys::Function,
    ) -> Result<String, JsValue> {
        // TEAM-286: Convert JS operation to Rust
        let op: Operation = js_to_operation(operation)?;

        // TEAM-286: Clone callback for use in closure
        let callback = on_line.clone();

        // TEAM-286: Use existing job-client!
        let job_id = self.inner
            .submit_and_stream(op, move |line| {
                // Call JavaScript callback
                let this = JsValue::null();
                let line_js = JsValue::from_str(line);
                
                // Ignore callback errors (non-critical)
                let _ = callback.call1(&this, &line_js);
                
                Ok(())
            })
            .await
            .map_err(error_to_js)?;

        Ok(job_id)
    }

    /// Submit a job without streaming
    ///
    /// TEAM-286: Wraps JobClient::submit()
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const jobId = await client.submit({ operation: 'status' });
    /// ```
    #[wasm_bindgen]
    pub async fn submit(&self, operation: JsValue) -> Result<String, JsValue> {
        let op: Operation = js_to_operation(operation)?;
        
        self.inner
            .submit(op)
            .await
            .map_err(error_to_js)
    }

    /// Health check
    ///
    /// TEAM-286: Simple GET /health request
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const health = await client.health();
    /// console.log(health); // { status: "ok" }
    /// ```
    #[wasm_bindgen]
    pub async fn health(&self) -> Result<JsValue, JsValue> {
        // TEAM-286: Use web_sys to make HTTP request
        use web_sys::{Request, RequestInit, Response};

        let url = format!("{}/health", self.inner.base_url());
        
        let mut opts = RequestInit::new();
        opts.method("GET");

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to create request: {:?}", e)))?;

        // Get window object
        let window = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object"))?;

        // Make request
        let resp_value = wasm_bindgen_futures::JsFuture::from(
            window.fetch_with_request(&request)
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("Fetch failed: {:?}", e)))?;

        let resp: Response = resp_value.dyn_into()
            .map_err(|_| JsValue::from_str("Response is not a Response object"))?;

        // Parse JSON
        let json = wasm_bindgen_futures::JsFuture::from(
            resp.json()
                .map_err(|e| JsValue::from_str(&format!("Failed to get JSON: {:?}", e)))?
        )
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to parse JSON: {:?}", e)))?;

        Ok(json)
    }
}
```

---

### Step 2: Operation Builders (2 hours)

**File:** `consumers/rbee-sdk/src/operations.rs`

```rust
// TEAM-286: Operation builder helpers for JavaScript

use wasm_bindgen::prelude::*;
use operations_contract::Operation;
use serde_wasm_bindgen;

/// Operation builders for JavaScript
///
/// TEAM-286: These provide convenient ways to construct operations
/// from JavaScript without manually building objects
#[wasm_bindgen]
pub struct OperationBuilder;

#[wasm_bindgen]
impl OperationBuilder {
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

    /// Create an Infer operation
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const op = OperationBuilder.infer({
    ///   model: 'llama-3-8b',
    ///   prompt: 'Hello!',
    ///   max_tokens: 100,
    ///   temperature: 0.7,
    ///   stream: true,
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn infer(params: JsValue) -> Result<JsValue, JsValue> {
        use operations_contract::InferRequest;
        
        // Parse JavaScript object to InferRequest
        let req: InferRequest = serde_wasm_bindgen::from_value(params)
            .map_err(|e| JsValue::from_str(&format!("Invalid infer params: {}", e)))?;

        // Create Operation::Infer
        let op = Operation::Infer(req);
        
        // Convert back to JavaScript
        Ok(serde_wasm_bindgen::to_value(&op).unwrap())
    }

    // TEAM-286: Add more builders in Phase 3
}
```

Add to `src/lib.rs`:
```rust
mod operations;
pub use operations::OperationBuilder;
```

---

### Step 3: Improved Type Conversions (1 hour)

**File:** `consumers/rbee-sdk/src/types.rs` (extend)

```rust
// TEAM-286: Enhanced type conversions

use wasm_bindgen::prelude::*;
use operations_contract::Operation;
use serde_wasm_bindgen;

/// Convert JavaScript operation to Rust Operation
pub fn js_to_operation(js_value: JsValue) -> Result<Operation, JsValue> {
    serde_wasm_bindgen::from_value(js_value)
        .map_err(|e| {
            JsValue::from_str(&format!(
                "Failed to parse operation. Expected format: {{ operation: 'status' }}\nError: {}",
                e
            ))
        })
}

/// Convert Rust operation to JavaScript
pub fn operation_to_js(op: &Operation) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(op)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize operation: {}", e)))
}

/// Convert Rust error to JavaScript error
pub fn error_to_js(error: anyhow::Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Create JavaScript error with stack trace
pub fn create_js_error(message: &str) -> JsValue {
    js_sys::Error::new(message).into()
}
```

---

### Step 4: Examples (2 hours)

**File:** `consumers/rbee-sdk/examples/basic.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>rbee SDK - Basic Example</title>
    <style>
        body { font-family: sans-serif; padding: 20px; }
        pre { background: #f5f5f5; padding: 10px; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>🐝 rbee SDK - Basic Example</h1>
    
    <button onclick="testHealth()">Test Health</button>
    <button onclick="testStatus()">Test Status</button>
    <button onclick="testStream()">Test Streaming</button>
    
    <div id="output"></div>
    
    <script type="module">
        import init, { RbeeClient, OperationBuilder } from '../pkg/rbee_sdk.js';
        
        // Initialize WASM
        await init();
        
        // Create client
        const client = new RbeeClient('http://localhost:8500');
        window.client = client;  // Make available to onclick handlers
        
        // Test health
        window.testHealth = async function() {
            const output = document.getElementById('output');
            output.innerHTML = '<p>Testing health...</p>';
            
            try {
                const health = await client.health();
                output.innerHTML = `<pre class="success">Health: ${JSON.stringify(health, null, 2)}</pre>`;
            } catch (error) {
                output.innerHTML = `<pre class="error">Error: ${error}</pre>`;
            }
        };
        
        // Test status operation
        window.testStatus = async function() {
            const output = document.getElementById('output');
            output.innerHTML = '<p>Getting status...</p>';
            
            try {
                const op = OperationBuilder.status();
                const lines = [];
                
                const jobId = await client.submitAndStream(op, (line) => {
                    lines.push(line);
                    output.innerHTML = `<pre>${lines.join('\n')}</pre>`;
                });
                
                output.innerHTML += `<p class="success">Job ID: ${jobId}</p>`;
            } catch (error) {
                output.innerHTML = `<pre class="error">Error: ${error}</pre>`;
            }
        };
        
        // Test streaming
        window.testStream = async function() {
            const output = document.getElementById('output');
            output.innerHTML = '<p>Starting inference...</p>';
            
            try {
                const op = OperationBuilder.infer({
                    hive_id: 'localhost',
                    model: 'llama-3-8b',
                    prompt: 'Write a haiku about WASM',
                    max_tokens: 50,
                    temperature: 0.7,
                    stream: true,
                });
                
                let fullText = '';
                
                const jobId = await client.submitAndStream(op, (line) => {
                    if (!line.includes('[DONE]')) {
                        fullText += line;
                        output.innerHTML = `<pre>${fullText}</pre>`;
                    }
                });
                
                output.innerHTML += `<p class="success">Complete! Job ID: ${jobId}</p>`;
            } catch (error) {
                output.innerHTML = `<pre class="error">Error: ${error}</pre>`;
            }
        };
        
        console.log('✅ rbee SDK loaded!');
        console.log('Client:', client);
    </script>
</body>
</html>
```

**File:** `consumers/rbee-sdk/examples/nodejs-basic.js`

```javascript
// TEAM-286: Node.js example using rbee SDK

// Use the nodejs build
const { RbeeClient, OperationBuilder } = require('../pkg/nodejs/rbee_sdk.js');

async function main() {
    const client = new RbeeClient('http://localhost:8500');
    
    console.log('🐝 Testing rbee SDK in Node.js\n');
    
    // Test health
    console.log('1. Testing health...');
    try {
        const health = await client.health();
        console.log('✅ Health:', health);
    } catch (error) {
        console.error('❌ Health failed:', error);
    }
    
    // Test status
    console.log('\n2. Testing status...');
    try {
        const op = OperationBuilder.status();
        const lines = [];
        
        const jobId = await client.submitAndStream(op, (line) => {
            lines.push(line);
            console.log('  ', line);
        });
        
        console.log('✅ Job ID:', jobId);
    } catch (error) {
        console.error('❌ Status failed:', error);
    }
}

main().catch(console.error);
```

---

### Step 5: Testing (2 hours)

**File:** `consumers/rbee-sdk/tests/integration.rs`

```rust
// TEAM-286: Integration tests for WASM bindings

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use rbee_sdk::{RbeeClient, OperationBuilder};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_client_creation() {
    let client = RbeeClient::new("http://localhost:8500".to_string());
    assert_eq!(client.base_url(), "http://localhost:8500");
}

#[wasm_bindgen_test]
fn test_operation_builders() {
    // Test that builders return valid JsValue
    let status = OperationBuilder::status();
    assert!(!status.is_undefined());
    
    let hive_list = OperationBuilder::hive_list();
    assert!(!hive_list.is_undefined());
}

// TEAM-286: Add async tests when queen-rbee is available
// #[wasm_bindgen_test]
// async fn test_health() {
//     let client = RbeeClient::new("http://localhost:8500".to_string());
//     let result = client.health().await;
//     assert!(result.is_ok());
// }
```

**Run tests:**
```bash
wasm-pack test --headless --firefox
```

---

## Verification Checklist

After completing Phase 2:

- [ ] submit_and_stream() works in browser
- [ ] submit_and_stream() works in Node.js
- [ ] JavaScript callbacks receive lines
- [ ] [DONE] marker is detected
- [ ] Operations are properly converted
- [ ] OperationBuilder works
- [ ] health() endpoint works
- [ ] Examples run successfully
- [ ] Tests pass
- [ ] All TEAM-286 signatures added

---

## Common Issues

### Issue: "Cannot call Rust async function from JavaScript"

**Fix:** Ensure using wasm-bindgen-futures:
```rust
use wasm_bindgen_futures::spawn_local;
```

### Issue: "Callback not called"

**Fix:** Check that callback is cloned before moving into closure:
```rust
let callback = on_line.clone();
```

### Issue: "ReferenceError: TextDecoder is not defined"

**Fix:** Add polyfill for Node.js or use newer Node version (18+)

---

## Next Phase

**Phase 3: All Operations** - Expose all remaining operations from operations-contract.

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Estimated Duration:** 2 days
