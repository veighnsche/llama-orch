# TEAM-286: job-client WASM Feature Flag Implementation

**Date:** Oct 24, 2025  
**Goal:** Make job-client work in both native Rust and WASM environments

---

## Strategy

Add a `wasm` feature flag that switches between:
- **Native:** `reqwest` (current implementation)
- **WASM:** `web-sys` (browser APIs)

---

## Implementation Plan

### 1. Update Cargo.toml

```toml
[package]
name = "job-client"
version.workspace = true
edition.workspace = true

[features]
default = []
wasm = ["wasm-bindgen", "wasm-bindgen-futures", "web-sys", "js-sys"]

[dependencies]
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
operations-contract = { path = "../../97_contracts/operations-contract" }
observability-narration-core = { path = "../narration-core" }

# Native dependencies (default)
reqwest = { version = "0.12", features = ["json", "stream"], optional = true }
futures = { version = "0.3", optional = true }

# WASM dependencies (feature = "wasm")
wasm-bindgen = { version = "0.2", optional = true }
wasm-bindgen-futures = { version = "0.4", optional = true }
web-sys = { version = "0.3", optional = true, features = [
    "Window",
    "Request",
    "RequestInit",
    "RequestMode",
    "Response",
    "Headers",
    "ReadableStream",
] }
js-sys = { version = "0.3", optional = true }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
reqwest = { version = "0.12", features = ["json", "stream"] }
futures = "0.3"

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = [
    "Window",
    "Request",
    "RequestInit",
    "RequestMode",
    "Response",
    "Headers",
    "ReadableStream",
] }
js-sys = "0.3"
```

### 2. Split Implementation

**File structure:**
```
job-client/
├── src/
│   ├── lib.rs          # Public API (same for both)
│   ├── native.rs       # reqwest implementation
│   └── wasm.rs         # web-sys implementation
```

### 3. lib.rs (Conditional Compilation)

```rust
//! Shared HTTP client for job submission and SSE streaming
//!
//! TEAM-286: Works in both native Rust and WASM environments

use anyhow::Result;
use operations_contract::Operation;

// Conditional module imports
#[cfg(not(target_arch = "wasm32"))]
mod native;

#[cfg(target_arch = "wasm32")]
mod wasm;

// Re-export the appropriate implementation
#[cfg(not(target_arch = "wasm32"))]
pub use native::JobClient;

#[cfg(target_arch = "wasm32")]
pub use wasm::JobClient;
```

### 4. native.rs (Current Implementation)

Just move the current code to `native.rs`:

```rust
// TEAM-286: Native implementation using reqwest

use anyhow::Result;
use futures::stream::StreamExt;
use operations_contract::Operation;

#[derive(Debug, Clone)]
pub struct JobClient {
    base_url: String,
    client: reqwest::Client,
}

impl JobClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { 
            base_url: base_url.into(), 
            client: reqwest::Client::new() 
        }
    }

    pub async fn submit_and_stream<F>(
        &self,
        operation: Operation,
        mut line_handler: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
    {
        // ... current implementation ...
    }

    pub async fn submit(&self, operation: Operation) -> Result<String> {
        // ... current implementation ...
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}
```

### 5. wasm.rs (New WASM Implementation)

```rust
// TEAM-286: WASM implementation using web-sys

use anyhow::Result;
use operations_contract::Operation;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

#[derive(Debug, Clone)]
pub struct JobClient {
    base_url: String,
}

impl JobClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self { 
            base_url: base_url.into(),
        }
    }

    pub async fn submit_and_stream<F>(
        &self,
        operation: Operation,
        mut line_handler: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
    {
        // 1. Serialize operation
        let payload = serde_json::to_string(&operation)?;

        // 2. POST to /v1/jobs
        let mut opts = RequestInit::new();
        opts.method("POST");
        opts.mode(RequestMode::Cors);
        opts.body(Some(&JsValue::from_str(&payload)));

        let url = format!("{}/v1/jobs", self.base_url);
        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| anyhow::anyhow!("Failed to create request: {:?}", e))?;

        request.headers().set("Content-Type", "application/json")
            .map_err(|e| anyhow::anyhow!("Failed to set header: {:?}", e))?;

        // Get window and fetch
        let window = web_sys::window()
            .ok_or_else(|| anyhow::anyhow!("No window object"))?;

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow::anyhow!("Fetch failed: {:?}", e))?;

        let resp: Response = resp_value.dyn_into()
            .map_err(|_| anyhow::anyhow!("Response is not a Response object"))?;

        // 3. Extract job_id
        let json = JsFuture::from(resp.json()
            .map_err(|e| anyhow::anyhow!("Failed to get JSON: {:?}", e))?)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {:?}", e))?;

        let job_data: serde_json::Value = serde_wasm_bindgen::from_value(json)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize: {}", e))?;

        let job_id = job_data
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No job_id in response"))?
            .to_string();

        // 4. For SSE streaming in WASM, we need to use EventSource
        // But that requires a different callback mechanism
        // For now, we'll use fetch with ReadableStream
        
        let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
        let stream_request = Request::new_with_str(&stream_url)
            .map_err(|e| anyhow::anyhow!("Failed to create stream request: {:?}", e))?;

        let stream_resp_value = JsFuture::from(window.fetch_with_request(&stream_request))
            .await
            .map_err(|e| anyhow::anyhow!("Stream fetch failed: {:?}", e))?;

        let stream_resp: Response = stream_resp_value.dyn_into()
            .map_err(|_| anyhow::anyhow!("Stream response is not a Response object"))?;

        // Get response text (simpler than streaming for now)
        let text_promise = stream_resp.text()
            .map_err(|e| anyhow::anyhow!("Failed to get text: {:?}", e))?;

        let text_value = JsFuture::from(text_promise)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read text: {:?}", e))?;

        let text = text_value.as_string()
            .ok_or_else(|| anyhow::anyhow!("Text is not a string"))?;

        // Process lines
        for line in text.lines() {
            let data = line.strip_prefix("data: ").unwrap_or(line);
            if data.is_empty() {
                continue;
            }
            line_handler(data)?;
            if data.contains("[DONE]") {
                break;
            }
        }

        Ok(job_id)
    }

    pub async fn submit(&self, operation: Operation) -> Result<String> {
        // Similar to submit_and_stream but without streaming
        let payload = serde_json::to_string(&operation)?;

        let mut opts = RequestInit::new();
        opts.method("POST");
        opts.mode(RequestMode::Cors);
        opts.body(Some(&JsValue::from_str(&payload)));

        let url = format!("{}/v1/jobs", self.base_url);
        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| anyhow::anyhow!("Failed to create request: {:?}", e))?;

        request.headers().set("Content-Type", "application/json")
            .map_err(|e| anyhow::anyhow!("Failed to set header: {:?}", e))?;

        let window = web_sys::window()
            .ok_or_else(|| anyhow::anyhow!("No window object"))?;

        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| anyhow::anyhow!("Fetch failed: {:?}", e))?;

        let resp: Response = resp_value.dyn_into()
            .map_err(|_| anyhow::anyhow!("Response is not a Response object"))?;

        let json = JsFuture::from(resp.json()
            .map_err(|e| anyhow::anyhow!("Failed to get JSON: {:?}", e))?)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse JSON: {:?}", e))?;

        let job_data: serde_json::Value = serde_wasm_bindgen::from_value(json)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize: {}", e))?;

        let job_id = job_data
            .get("job_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No job_id in response"))?
            .to_string();

        Ok(job_id)
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}
```

---

## Usage in rbee-sdk

```toml
# consumers/rbee-sdk/Cargo.toml

[dependencies]
job-client = { path = "../../bin/99_shared_crates/job-client" }
# No need to specify features - it auto-detects WASM target!
```

The `job-client` will automatically use:
- `native.rs` when compiling for native targets
- `wasm.rs` when compiling for `wasm32-unknown-unknown`

---

## Benefits

✅ **Single crate** - One job-client for all targets  
✅ **Automatic** - No manual feature selection needed  
✅ **Maintainable** - Shared API, different implementations  
✅ **Type-safe** - Same interface for both  
✅ **Reusable** - rbee-sdk just uses job-client normally  

---

## Timeline

1. Update Cargo.toml (10 min)
2. Create native.rs (5 min - just move code)
3. Create wasm.rs (2 hours - implement with web-sys)
4. Update lib.rs (5 min - conditional exports)
5. Test native build (5 min)
6. Test WASM build (10 min)

**Total:** ~2.5 hours

---

## Result

After this, rbee-sdk can simply use job-client and it will work in WASM!

```rust
// rbee-sdk/src/client.rs - NO CHANGES NEEDED!
use job_client::JobClient;

// This just works in WASM now!
let client = JobClient::new(base_url);
client.submit_and_stream(op, handler).await?;
```

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Status:** Ready to implement
