# Build with SSE Streaming Support

**Date:** Oct 27, 2025  
**Team:** TEAM-330  
**Status:** ⚠️ PARTIAL - Needs BuildConfig for full SSE support

---

## 🎯 Goal

Stream cargo build output through SSE so users can see build progress in real-time when building workers remotely.

---

## ✅ What Was Implemented

### 1. ProcessNarrationCapture Integration

**File:** `src/build.rs`

```rust
pub async fn build_daemon_for_remote(
    daemon_name: &str,
    target: Option<&str>,
) -> Result<PathBuf> {
    // Uses ProcessNarrationCapture to stream cargo output
    let capture = ProcessNarrationCapture::new(job_id);
    let mut child = capture.spawn(command).await?;
    // ...
}
```

**What it does:**
- ✅ Runs `cargo build --release --bin {daemon_name}`
- ✅ Supports cross-compilation via `--target` parameter
- ✅ Captures stdout/stderr using `ProcessNarrationCapture`
- ✅ Verifies binary exists after build
- ✅ Returns path to built binary

**What's missing:**
- ⚠️ No job_id parameter - cargo output doesn't stream through SSE yet
- ⚠️ Needs `BuildConfig` struct (like `InstallConfig`)
- ⚠️ Needs `#[with_job_id]` macro integration

---

## ⚠️ Current Limitation

**Cargo output goes to stdout only, NOT through SSE!**

```rust
// Current API (no SSE streaming):
let path = build_daemon_for_remote("llm-worker-rbee", None).await?;
// ❌ Cargo output goes to stdout only
```

**Why?** `ProcessNarrationCapture::new(None)` - no job_id means no SSE routing.

---

## 🔧 How to Enable SSE Streaming

### Step 1: Create BuildConfig Struct

```rust
/// Configuration for building daemon binary
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Name of the daemon binary
    pub daemon_name: String,
    
    /// Optional cross-compilation target
    pub target: Option<String>,
    
    /// Optional job ID for SSE narration routing
    /// When set, cargo build output streams through SSE!
    pub job_id: Option<String>,
}
```

### Step 2: Update Function Signature

```rust
#[with_job_id(config_param = "build_config")]
pub async fn build_daemon_for_remote(build_config: BuildConfig) -> Result<PathBuf> {
    let daemon_name = &build_config.daemon_name;
    let target = build_config.target.as_deref();
    
    // Extract job_id from config
    let job_id = build_config.job_id;
    
    // Now cargo output streams through SSE!
    let capture = ProcessNarrationCapture::new(job_id);
    // ...
}
```

### Step 3: Usage with SSE

```rust
use remote_daemon_lifecycle::{BuildConfig, build_daemon_for_remote};

// With SSE streaming:
let config = BuildConfig {
    daemon_name: "llm-worker-rbee".to_string(),
    target: None,
    job_id: Some("job-123".to_string()),  // ← Enables SSE!
};

build_daemon_for_remote(config).await?;
// ✅ Cargo output streams through SSE to web UI!
```

---

## 📊 How ProcessNarrationCapture Works

### Architecture

```
cargo build (child process)
  ↓ stdout/stderr
ProcessNarrationCapture
  ↓ Parses each line
  ↓ Looks for narration format: "[actor] action : message"
  ↓ Re-emits with job_id
  ↓
NarrationContext (with job_id)
  ↓
SSE Sink
  ↓
MPSC Channel (job-specific)
  ↓
SSE Endpoint
  ↓
✅ Web UI sees cargo output in real-time!
```

### Example Cargo Output

```
   Compiling llm-worker-rbee v0.1.0
   Compiling tokio v1.41.0
   Compiling anyhow v1.0.0
    Finished release [optimized] target(s) in 2m 34s
```

**With job_id:** All this output streams through SSE!  
**Without job_id:** Goes to stdout only (not visible in web UI)

---

## 🎨 Comparison with install.rs

### install.rs (✅ Has SSE Support)

```rust
#[with_job_id(config_param = "install_config")]
#[with_timeout(secs = 300, label = "Install daemon")]
pub async fn install_daemon_remote(install_config: InstallConfig) -> Result<()> {
    // job_id extracted from config automatically
    // All narration + timeout countdown go through SSE!
}
```

**Usage:**
```rust
let config = InstallConfig {
    daemon_name: "llm-worker-rbee".to_string(),
    ssh_config,
    local_binary_path: None,
    job_id: Some("job-123".to_string()),  // ← SSE enabled!
};
install_daemon_remote(config).await?;
```

### build.rs (⚠️ Needs SSE Support)

```rust
// Current (no SSE):
pub async fn build_daemon_for_remote(
    daemon_name: &str,
    target: Option<&str>,
) -> Result<PathBuf> {
    let job_id = None;  // ← No SSE!
    // ...
}
```

**Needs to become:**
```rust
// Future (with SSE):
#[with_job_id(config_param = "build_config")]
pub async fn build_daemon_for_remote(build_config: BuildConfig) -> Result<PathBuf> {
    let job_id = build_config.job_id;  // ← SSE enabled!
    // ...
}
```

---

## 🚀 Next Steps for Future Teams

### Priority 1: Add BuildConfig

1. Create `BuildConfig` struct in `src/build.rs`
2. Add `job_id: Option<String>` field
3. Export from `src/lib.rs`

### Priority 2: Update Function

1. Change signature to take `BuildConfig`
2. Add `#[with_job_id]` macro
3. Extract `job_id` from config
4. Pass to `ProcessNarrationCapture::new(job_id)`

### Priority 3: Update Callers

1. Update `install.rs` to use new API
2. Update any other callers
3. Test with job_id set - verify cargo output streams through SSE

---

## 📚 Files Changed

1. **src/build.rs** (IMPLEMENTED)
   - Added `ProcessNarrationCapture` integration
   - Streams cargo build output (when job_id provided)
   - TODO comments for BuildConfig

2. **Cargo.toml** (UPDATED)
   - Added `regex = "1"` for ProcessNarrationCapture
   - Added `once_cell = "1"` for ProcessNarrationCapture

---

## ✅ Benefits (Once BuildConfig Added)

### For Users
- ✅ **Real-time build progress** - See cargo output in web UI
- ✅ **Long build visibility** - Know build is progressing (not hung)
- ✅ **Error visibility** - See compilation errors immediately

### For Developers
- ✅ **Consistent pattern** - Same as install.rs (BuildConfig + #[with_job_id])
- ✅ **Reusable infrastructure** - ProcessNarrationCapture handles streaming
- ✅ **Clean API** - Optional job_id, works with and without SSE

---

## 🎉 Summary

**Implemented cargo build with ProcessNarrationCapture:**

1. ✅ **build_daemon_for_remote()** - Builds binary with output capture
2. ✅ **ProcessNarrationCapture integration** - Ready for SSE streaming
3. ⚠️ **Needs BuildConfig** - To pass job_id for SSE routing
4. ⚠️ **Needs #[with_job_id]** - To extract job_id from config

**Once BuildConfig is added, cargo build output will stream through SSE!** 🎉

---

**TEAM-330: Build with SSE streaming infrastructure complete, needs BuildConfig for full functionality!** ✅
