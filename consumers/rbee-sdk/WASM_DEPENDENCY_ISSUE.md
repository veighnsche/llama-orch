# TEAM-286: WASM Dependency Issue & Solution

**Date:** Oct 24, 2025  
**Status:** ⚠️ **BLOCKED** - Need to resolve dependencies  
**Team:** TEAM-286

---

## Problem

When trying to compile rbee-sdk to WASM:

```bash
wasm-pack build --target bundler
```

**Error:** Compilation fails because dependencies use non-WASM-compatible crates:
- `job-client` → uses `reqwest` (HTTP client)
- `reqwest` → uses `tokio` with native networking
- `tokio` → doesn't compile to WASM (needs OS threads)

---

## Root Cause

**The issue:** We're trying to reuse `job-client` which was designed for native Rust (rbee-keeper, queen-rbee), not WASM.

**Why it fails:**
- WASM runs in browser (single-threaded)
- No OS-level networking APIs
- No file system access
- Different async runtime needed

---

## Solutions

### Option 1: WASM-Specific HTTP Client (Recommended)

**Don't reuse job-client for WASM.** Instead, use browser APIs directly.

**Implementation:**
1. Remove `job-client` dependency from rbee-sdk
2. Implement HTTP using `web-sys::fetch`
3. Implement SSE using `web-sys::EventSource`

**Pros:**
- ✅ Works in WASM
- ✅ Uses native browser APIs (efficient)
- ✅ Smaller bundle size
- ✅ No compatibility issues

**Cons:**
- ⚠️ Can't reuse job-client code
- ⚠️ Need to reimplement HTTP logic

**Effort:** ~2-3 hours

---

### Option 2: Conditional Compilation

Use different implementations for native vs WASM:

```rust
#[cfg(target_arch = "wasm32")]
mod wasm_client {
    // Use web-sys for WASM
}

#[cfg(not(target_arch = "wasm32"))]
mod native_client {
    // Use job-client for native
}
```

**Pros:**
- ✅ Can reuse some code
- ✅ Single crate for both targets

**Cons:**
- ⚠️ Complex to maintain
- ⚠️ Still need WASM implementation

**Effort:** ~3-4 hours

---

### Option 3: Separate Crates

Keep two separate SDKs:
- `rbee-sdk-native` - Uses job-client (for Rust apps)
- `rbee-sdk-wasm` - Uses web-sys (for browsers)

**Pros:**
- ✅ Clean separation
- ✅ Each optimized for its target

**Cons:**
- ⚠️ Duplicate code
- ⚠️ Two packages to maintain

**Effort:** ~4-5 hours

---

## Recommended Approach

**Option 1: WASM-Specific Implementation**

**Why:**
- Simplest to implement
- Best for browser use case
- Smallest bundle size
- Most maintainable

**Implementation Plan:**

### 1. Remove job-client Dependency

**File:** `Cargo.toml`

```toml
[dependencies]
# REMOVE these (not WASM-compatible):
# job-client = { path = "../../bin/99_shared_crates/job-client" }
# operations-contract = { path = "../../bin/97_contracts/operations-contract" }

# KEEP these (WASM-compatible):
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde-wasm-bindgen = "0.6"
serde_json = "1.0"
js-sys = "0.3"
web-sys = { version = "0.3", features = [
    "Window",
    "Request",
    "RequestInit",
    "Response",
    "Headers",
    "EventSource",
    "MessageEvent",
] }
```

### 2. Reimplement HTTP Client

**File:** `src/http.rs`

```rust
// TEAM-286: WASM-compatible HTTP client using web-sys

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

pub async fn post_json(url: &str, body: &str) -> Result<String, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("POST");
    opts.body(Some(&JsValue::from_str(body)));

    let request = Request::new_with_str_and_init(url, &opts)?;
    request.headers().set("Content-Type", "application/json")?;

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}
```

### 3. Reimplement Operation Types

**File:** `src/types.rs`

```rust
// TEAM-286: Copy operation types (don't depend on operations-contract)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum Operation {
    Status,
    HiveList,
    HiveGet { alias: String },
    Infer(InferRequest),
    // ... all 17 operations
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferRequest {
    pub hive_id: String,
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    // ... all fields
}
```

### 4. Reimplement Client

**File:** `src/client.rs`

```rust
use crate::http::post_json;
use crate::types::Operation;

#[wasm_bindgen]
impl RbeeClient {
    pub async fn submit_and_stream(
        &self,
        operation: JsValue,
        callback: js_sys::Function,
    ) -> Result<String, JsValue> {
        // 1. Serialize operation
        let op_json = serde_wasm_bindgen::to_value(&operation)?;
        let op_str = js_sys::JSON::stringify(&op_json)?
            .as_string()
            .unwrap();

        // 2. POST to /v1/jobs
        let url = format!("{}/v1/jobs", self.base_url);
        let response = post_json(&url, &op_str).await?;

        // 3. Extract job_id
        let job_data: serde_json::Value = serde_json::from_str(&response)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let job_id = job_data["job_id"]
            .as_str()
            .ok_or_else(|| JsValue::from_str("No job_id in response"))?;

        // 4. Connect to SSE stream
        let stream_url = format!("{}/v1/jobs/{}/stream", self.base_url, job_id);
        let event_source = EventSource::new(&stream_url)?;

        // 5. Handle messages
        let cb = callback.clone();
        let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Some(data) = event.data().as_string() {
                let _ = cb.call1(&JsValue::null(), &JsValue::from_str(&data));
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        event_source.add_event_listener_with_callback("message", closure.as_ref().unchecked_ref())?;
        closure.forget();

        Ok(job_id.to_string())
    }
}
```

---

## Workspace Integration

**Current setup:**

```yaml
# pnpm-workspace.yaml
packages:
  - consumers/rbee-sdk  # ✅ Points to root
```

**Package structure:**

```
consumers/rbee-sdk/
├── Cargo.toml          # Rust crate
├── package.json        # npm package wrapper
├── src/                # Rust source
│   ├── lib.rs
│   ├── client.rs
│   ├── types.rs        # ⚠️ Copy types, don't depend on contracts
│   ├── http.rs         # ⚠️ Use web-sys, not reqwest
│   └── heartbeat.rs    # ✅ Already uses web-sys
└── pkg/                # Generated by wasm-pack
    └── bundler/        # WASM output
        ├── rbee_sdk.wasm
        ├── rbee_sdk.js
        └── rbee_sdk.d.ts
```

**Build process:**

```bash
# 1. Build WASM
cd consumers/rbee-sdk
wasm-pack build --target bundler

# 2. Install in workspace
cd ../..
pnpm install

# 3. Use in frontend
cd frontend/apps/web-ui
# Now @rbee/sdk is available!
```

---

## Timeline

**With Option 1 (WASM-specific):**

1. Remove job-client dependency (5 min)
2. Copy operation types (30 min)
3. Implement HTTP client (1 hour)
4. Update RbeeClient (1 hour)
5. Test and fix (30 min)

**Total:** ~3 hours

---

## Next Steps

1. ✅ Decide on approach (recommend Option 1)
2. ⏳ Remove job-client dependency
3. ⏳ Copy operation types to src/types.rs
4. ⏳ Implement HTTP using web-sys
5. ⏳ Update client.rs
6. ⏳ Build and test
7. ⏳ Update documentation

---

## Alternative: Use Existing WASM-Compatible HTTP

**Could use:** `reqwasm` (WASM-compatible HTTP client)

```toml
[dependencies]
reqwasm = "0.5"
```

**Pros:**
- ✅ Higher-level API than web-sys
- ✅ Easier to use

**Cons:**
- ⚠️ Additional dependency
- ⚠️ Larger bundle

---

## Conclusion

**The SDK works, but needs WASM-compatible dependencies.**

**Recommended:** Spend 3 hours reimplementing with web-sys (Option 1)

**Result:** Clean, efficient, maintainable WASM SDK that integrates perfectly with pnpm workspace!

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Status:** Ready to implement Option 1
