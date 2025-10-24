# TEAM-286: narration-core WASM Compatibility Analysis

**Date:** Oct 24, 2025  
**Question:** Can we make narration-core WASM-compatible?  
**Answer:** ⚠️ **Partially - with significant limitations**

---

## Current Dependencies

### Core Dependencies
```toml
tracing = { workspace = true }           # ✅ WASM-compatible
serde = { workspace = true }             # ✅ WASM-compatible
serde_json = { workspace = true }        # ✅ WASM-compatible
regex = { workspace = true }             # ✅ WASM-compatible
uuid = { workspace = true }              # ✅ WASM-compatible
once_cell = "1.19"                       # ✅ WASM-compatible
```

### Problematic Dependencies
```toml
tokio = { workspace = true, features = ["sync"] }  # ⚠️ LIMITED WASM support
```

---

## Tokio WASM Support Status

### What Works on WASM (wasm32-unknown-unknown)

**Limited features only:**
- ✅ `tokio::sync::mpsc` - Multi-producer, single-consumer channels
- ✅ `tokio::sync::Mutex` - Async mutex
- ✅ `tokio::sync::RwLock` - Async read-write lock
- ✅ `tokio::sync::Semaphore` - Async semaphore
- ⚠️ `tokio::task_local!` - **DOES NOT WORK** on wasm32-unknown-unknown

### What Doesn't Work on WASM

❌ **Runtime features:**
- `tokio::runtime` - No multi-threaded runtime
- `tokio::spawn` - No task spawning
- `tokio::task_local!` - No task-local storage

❌ **I/O features:**
- `tokio::net` - No networking (use browser fetch API)
- `tokio::fs` - No file system
- `tokio::process` - No process spawning

❌ **Time features:**
- `tokio::time` - Limited/no timer support

---

## narration-core WASM Blockers

### 1. `tokio::task_local!` (CRITICAL)

**File:** `src/context.rs`

```rust
tokio::task_local! {
    static NARRATION_CONTEXT: RefCell<NarrationContext>;
}
```

**Problem:** `task_local!` doesn't work on wasm32-unknown-unknown because:
- WASM is single-threaded
- No OS-level thread-local storage
- Browser environment doesn't support task-local state

**Impact:** The entire `NarrationContext` system won't work in WASM.

### 2. `tokio::sync::mpsc` (WORKS)

**File:** `src/sse_sink.rs`

```rust
use tokio::sync::mpsc;

type JobSender = mpsc::Sender<NarrationEvent>;
type JobReceiver = mpsc::Receiver<NarrationEvent>;
```

**Status:** ✅ This DOES work on WASM!
- `tokio::sync` is supported on wasm32-unknown-unknown
- MPSC channels work fine

---

## Solutions

### Option 1: Make narration-core Optional (Recommended)

**Don't use narration-core in WASM.** It's designed for server-side observability.

**Implementation:**
```toml
# job-client/Cargo.toml

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
observability-narration-core = { path = "../narration-core" }
```

**Code:**
```rust
// job-client/src/lib.rs

#[cfg(not(target_arch = "wasm32"))]
use observability_narration_core as narration;

#[cfg(not(target_arch = "wasm32"))]
fn emit_narration(msg: &str) {
    narration::narrate!(/* ... */);
}

#[cfg(target_arch = "wasm32")]
fn emit_narration(_msg: &str) {
    // No-op in WASM
}
```

**Pros:**
- ✅ Simple
- ✅ No code changes to narration-core
- ✅ Works immediately

**Cons:**
- ⚠️ No narration in WASM (but do you need it?)

---

### Option 2: WASM-Compatible Subset

Create a minimal WASM-compatible version with feature flags.

**Changes needed:**

#### 1. Add WASM feature flag

```toml
# narration-core/Cargo.toml

[features]
default = ["tokio-runtime"]
tokio-runtime = []  # Full tokio features (native only)
wasm = []           # WASM-compatible subset

[dependencies]
tokio = { version = "1", features = ["sync"], optional = true }

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tokio = { version = "1", features = ["sync"] }
```

#### 2. Replace `task_local!` with thread-local

```rust
// src/context.rs

#[cfg(not(target_arch = "wasm32"))]
tokio::task_local! {
    static NARRATION_CONTEXT: RefCell<NarrationContext>;
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static NARRATION_CONTEXT: RefCell<NarrationContext> = RefCell::new(NarrationContext::default());
}
```

#### 3. Conditional async context

```rust
// src/context.rs

#[cfg(not(target_arch = "wasm32"))]
pub async fn with_narration_context<F>(ctx: NarrationContext, f: F) -> F::Output
where
    F: std::future::Future,
{
    NARRATION_CONTEXT.scope(RefCell::new(ctx), f).await
}

#[cfg(target_arch = "wasm32")]
pub async fn with_narration_context<F>(ctx: NarrationContext, f: F) -> F::Output
where
    F: std::future::Future,
{
    NARRATION_CONTEXT.with(|c| *c.borrow_mut() = ctx);
    let result = f.await;
    NARRATION_CONTEXT.with(|c| *c.borrow_mut() = NarrationContext::default());
    result
}
```

**Pros:**
- ✅ Narration works in WASM
- ✅ Same API

**Cons:**
- ⚠️ Significant code changes
- ⚠️ Different semantics (thread-local vs task-local)
- ⚠️ Maintenance burden

**Effort:** ~4-6 hours

---

### Option 3: Browser-Native Logging

For WASM, use browser console instead of narration.

```rust
#[cfg(target_arch = "wasm32")]
fn log(msg: &str) {
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(msg));
}

#[cfg(not(target_arch = "wasm32"))]
fn log(msg: &str) {
    narration::narrate!(/* ... */);
}
```

**Pros:**
- ✅ Simple
- ✅ Uses native browser tools
- ✅ No dependencies

**Cons:**
- ⚠️ Different API
- ⚠️ No structured fields

---

## Recommendation

### For job-client: Option 1 (Make it Optional)

**Why:**
1. **job-client is used in both native and WASM**
2. **Narration is for server-side observability** (not needed in browser)
3. **Simple conditional compilation** (no code changes to narration-core)

**Implementation:**

```toml
# job-client/Cargo.toml

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
observability-narration-core = { path = "../narration-core" }
```

```rust
// job-client/src/lib.rs

#[cfg(not(target_arch = "wasm32"))]
use observability_narration_core::narrate;

impl JobClient {
    pub async fn submit_and_stream<F>(&self, op: Operation, handler: F) -> Result<String> {
        #[cfg(not(target_arch = "wasm32"))]
        narrate!(/* ... */);
        
        // ... rest of implementation
    }
}
```

**Result:**
- ✅ Native: Full narration support
- ✅ WASM: No narration (uses browser console if needed)
- ✅ Zero changes to narration-core
- ✅ Works immediately

---

### For rbee-sdk: Use Browser Console

```rust
// rbee-sdk/src/client.rs

#[cfg(target_arch = "wasm32")]
fn log(msg: &str) {
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(msg));
}

#[wasm_bindgen]
impl RbeeClient {
    pub async fn submit_and_stream(&self, op: JsValue, callback: js_sys::Function) -> Result<String, JsValue> {
        log("Submitting job...");
        // ... implementation
    }
}
```

---

## Summary

### Can narration-core be WASM-compatible?

**Technically:** Yes, with Option 2 (4-6 hours of work)

**Practically:** No, it's not worth it because:
1. **Narration is for server-side observability** (logs, SSE streams)
2. **WASM runs in browser** (use browser console instead)
3. **job-client doesn't need narration in WASM** (just HTTP calls)

### Recommended Approach

✅ **Make narration-core optional for WASM targets**

**Implementation:**
1. Remove narration-core from job-client's core dependencies
2. Add it as target-specific dependency (native only)
3. Use `#[cfg(not(target_arch = "wasm32"))]` for narration calls
4. Use browser console for WASM logging if needed

**Effort:** ~30 minutes

**Result:** job-client works on both native and WASM, narration-core stays unchanged

---

## Next Steps

1. ✅ Keep narration-core as-is (no WASM support needed)
2. ✅ Make it optional in job-client (already done in TEAM-286)
3. ⏳ Add browser console logging to rbee-sdk if needed
4. ⏳ Document that narration is native-only

---

**Created by:** TEAM-286  
**Date:** Oct 24, 2025  
**Conclusion:** Don't make narration-core WASM-compatible. Use conditional compilation instead.
