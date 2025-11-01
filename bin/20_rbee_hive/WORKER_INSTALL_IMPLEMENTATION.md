# Worker Installation Implementation Plan

**Date:** 2025-11-01  
**Status:** 🚧 IN PROGRESS

## Overview

Implement full worker installation flow from UI to PKGBUILD execution.

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ FRONTEND (React)                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ WorkerCatalogView.tsx                                                │
│   ↓ onClick="Install Worker"                                        │
│ useHiveOperations.installWorker(workerId)                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ SDK (rbee-hive-sdk)                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ HiveOperations.installWorker(workerId)                              │
│   → Creates WorkerInstall operation                                 │
│   → Calls JobClient.submitAndStream()                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ JOB CLIENT (rbee-job-client)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ POST http://localhost:7835/v1/jobs                                  │
│ Body: { "operation": "worker_install", "worker_id": "..." }        │
│   → Returns job_id                                                  │
│   → Connects to SSE: /v1/jobs/{job_id}/stream                      │
│   → Streams output to callback                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ HIVE JOB ROUTER (job_router.rs)                                     │
├─────────────────────────────────────────────────────────────────────┤
│ route_operation()                                                    │
│   → Parses Operation::WorkerInstall                                 │
│   → Calls worker_install::handle_worker_install()                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ WORKER INSTALL HANDLER (worker_install.rs)                          │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Fetch worker metadata from catalog                               │
│    GET http://localhost:8787/workers/{worker_id}                    │
│                                                                      │
│ 2. Check platform compatibility                                     │
│    - Verify current platform matches worker.platforms               │
│    - Verify architecture matches worker.architectures               │
│                                                                      │
│ 3. Download PKGBUILD                                                │
│    GET http://localhost:8787/workers/{worker_id}/PKGBUILD           │
│                                                                      │
│ 4. Parse PKGBUILD                                                   │
│    PkgBuild::parse(content)                                         │
│    - Extract pkgname, pkgver, depends, makedepends                  │
│    - Extract build() and package() functions                        │
│                                                                      │
│ 5. Check dependencies                                               │
│    - Verify runtime deps (gcc, cuda, etc.)                          │
│    - Verify build deps (rust, cargo, etc.)                          │
│                                                                      │
│ 6. Create temp directories                                          │
│    - srcdir: /tmp/worker-install/{worker_id}/src                    │
│    - pkgdir: /tmp/worker-install/{worker_id}/pkg                    │
│    - workdir: /tmp/worker-install/{worker_id}                       │
│                                                                      │
│ 7. Execute build()                                                  │
│    PkgBuildExecutor::build(&pkgbuild, |line| {                      │
│        n!("build_output", "{}", line);  // Stream to SSE            │
│    })                                                                │
│                                                                      │
│ 8. Execute package()                                                │
│    PkgBuildExecutor::package(&pkgbuild, |line| {                    │
│        n!("package_output", "{}", line);  // Stream to SSE          │
│    })                                                                │
│                                                                      │
│ 9. Install binary                                                   │
│    - Copy from pkgdir/usr/local/bin/{binary_name}                   │
│    - To /usr/local/bin/{binary_name}                                │
│    - Set executable permissions                                     │
│                                                                      │
│ 10. Update capabilities cache                                       │
│     - Add installed worker to capabilities                          │
│     - Save capabilities.json                                        │
│                                                                      │
│ 11. Cleanup temp directories                                        │
│     - Remove /tmp/worker-install/{worker_id}                        │
└─────────────────────────────────────────────────────────────────────┘
```

## File Changes Required

### 1. Contracts (operations-contract)

**File:** `bin/97_contracts/operations-contract/src/lib.rs`
- ✅ Add `WorkerInstall(WorkerInstallRequest)` variant

**File:** `bin/97_contracts/operations-contract/src/requests.rs`
- ✅ Add `WorkerInstallRequest` struct

**File:** `bin/97_contracts/operations-contract/src/operation_impl.rs`
- ⏳ Add `WorkerInstall` case to `name()` method
- ⏳ Add `WorkerInstall` case to `target_server()` method

### 2. Hive Backend (rbee-hive)

**File:** `bin/20_rbee_hive/src/lib.rs`
- ⏳ Add `pub mod worker_install;`

**File:** `bin/20_rbee_hive/src/worker_install.rs` (NEW)
- ⏳ Create full implementation with PKGBUILD download + execution

**File:** `bin/20_rbee_hive/src/job_router.rs`
- ⏳ Add `Operation::WorkerInstall` match arm

**File:** `bin/20_rbee_hive/Cargo.toml`
- ⏳ Add `reqwest` dependency (for HTTP requests to catalog)
- ⏳ Add `tempfile` dependency (for temp directories)

### 3. SDK (rbee-hive-sdk)

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs`
- ⏳ Add `install_worker()` method

### 4. React Hooks (rbee-hive-react)

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useHiveOperations.ts`
- ⏳ Add `installWorker()` method
- ⏳ Add `installingWorker` state
- ⏳ Add `installError` state

### 5. UI Components

**File:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx`
- ✅ Already has `onInstall` handler
- ⏳ Wire to `useHiveOperations().installWorker()`

## Implementation Steps

### Step 1: Update operation_impl.rs ✅

Add WorkerInstall to name() and target_server() methods.

### Step 2: Create worker_install.rs 🚧

Full implementation with:
- Catalog API client
- PKGBUILD download
- Parser integration
- Executor integration
- Dependency checking
- Binary installation
- Capabilities update

### Step 3: Update job_router.rs ✅

Add match arm for WorkerInstall operation.

### Step 4: Update SDK ⏳

Add installWorker() method that creates WorkerInstall operation.

### Step 5: Update React Hooks ⏳

Add installWorker() hook with SSE streaming.

### Step 6: Wire UI ⏳

Connect WorkerCatalogView to useHiveOperations().installWorker().

## SSE Output Example

```
data: 📦 Fetching worker metadata from catalog...
data: ✓ Worker: llm-worker-rbee-cpu v0.1.0
data: ✓ Platform compatible: linux
data: ✓ Architecture compatible: x86_64
data: 📄 Downloading PKGBUILD...
data: ✓ PKGBUILD downloaded (2.1 KB)
data: 🔍 Parsing PKGBUILD...
data: ✓ Parsed: pkgname=llm-worker-rbee-cpu, pkgver=0.1.0
data: 🔧 Checking dependencies...
data: ✓ Runtime: gcc
data: ✓ Build: rust, cargo
data: 🏗️  Starting build phase...
data: Building llm-worker-rbee-cpu v0.1.0
data:    Compiling candle-core v0.3.0
data:    Compiling llm-worker-rbee v0.1.0
data:     Finished release [optimized] target(s) in 2m 34s
data: ✓ Build complete
data: 📦 Starting package phase...
data: Packaging llm-worker-rbee-cpu v0.1.0
data: Installing binary to /usr/local/bin/llm-worker-rbee-cpu
data: ✓ Package complete
data: 💾 Installing binary...
data: ✓ Binary installed: /usr/local/bin/llm-worker-rbee-cpu
data: 📝 Updating capabilities cache...
data: ✓ Capabilities updated
data: 🧹 Cleaning up temp files...
data: ✓ Cleanup complete
data: ✅ Worker installation complete!
data: [DONE]
```

## Error Handling

### Platform Incompatible
```
data: ❌ Platform incompatible
data: Worker requires: linux, macos
data: Current platform: windows
data: [ERROR] Cannot install worker on this platform
```

### Missing Dependencies
```
data: ❌ Missing dependencies
data: Runtime: gcc ✓, cuda ✗
data: Build: rust ✓, cargo ✓
data: [ERROR] Please install cuda before proceeding
```

### Build Failure
```
data: 🏗️  Starting build phase...
data: Building llm-worker-rbee-cuda v0.1.0
data: ERROR: CUDA toolkit not found
data: ❌ Build failed with exit code 1
data: [ERROR] Build phase failed
```

## Testing Plan

### Manual Test
```bash
# 1. Start catalog
cd bin/80-hono-worker-catalog
pnpm dev  # Port 8787

# 2. Start hive
cd bin/20_rbee_hive
cargo run  # Port 7835

# 3. Start UI
cd bin/20_rbee_hive/ui/app
pnpm dev  # Port 7836

# 4. Open browser
http://localhost:7836

# 5. Navigate to Worker Management → Worker Catalog

# 6. Click "Install Worker" on llm-worker-rbee-cpu

# 7. Watch SSE stream in UI

# 8. Verify binary installed
ls -la /usr/local/bin/llm-worker-rbee-cpu
```

### Integration Test
```rust
#[tokio::test]
async fn test_worker_install_flow() {
    // 1. Mock catalog server
    // 2. Submit WorkerInstall operation
    // 3. Verify PKGBUILD downloaded
    // 4. Verify build executed
    // 5. Verify binary installed
    // 6. Verify capabilities updated
}
```

## Next Steps

1. ✅ Add WorkerInstall operation to contracts
2. 🚧 Implement worker_install.rs handler
3. ⏳ Update job_router.rs
4. ⏳ Update SDK
5. ⏳ Update React hooks
6. ⏳ Wire UI
7. ⏳ Test end-to-end
8. ⏳ Add error handling
9. ⏳ Add progress indicators
10. ⏳ Document

---

**Status:** Contracts updated ✅, Implementation in progress 🚧
