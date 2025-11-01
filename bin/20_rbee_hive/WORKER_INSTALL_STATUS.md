# Worker Installation - Current Status

**Date:** 2025-11-01  
**Status:** 🎯 READY FOR IMPLEMENTATION

## ✅ Completed

### 1. PKGBUILD Infrastructure
- ✅ **Parser** (`pkgbuild_parser.rs`) - Parse PKGBUILD files
- ✅ **Executor** (`pkgbuild_executor.rs`) - Execute build() and package() functions
- ✅ **Tests** - 26 behavioral tests (10 executor + 16 parser)

### 2. Worker Catalog Service
- ✅ **CORS enabled** - Frontend can fetch from catalog
- ✅ **Modular structure** - Split into types, data, routes
- ✅ **3 workers** - CPU, CUDA, Metal variants
- ✅ **PKGBUILD serving** - All 3 PKGBUILD files available

### 3. UI Components
- ✅ **WorkerCatalogView** - Browse and install workers
- ✅ **Platform detection** - Filter incompatible workers
- ✅ **Rich metadata** - Dependencies, formats, capabilities
- ✅ **Install/Remove buttons** - UI ready for backend integration

### 4. Operation Contracts
- ✅ **WorkerInstall operation** - Added to Operation enum
- ✅ **WorkerInstallRequest** - Typed request structure
- ✅ **Capabilities endpoint** - `/v1/capabilities` exists

## 🚧 In Progress

### Backend Handler (worker_install.rs)

**Need to implement:**

```rust
// bin/20_rbee_hive/src/worker_install.rs

pub async fn handle_worker_install(
    worker_id: String,
) -> Result<()> {
    // 1. Fetch worker metadata from catalog
    n!("fetch_metadata", "📦 Fetching worker metadata...");
    let worker = fetch_worker_metadata(&worker_id).await?;
    
    // 2. Check platform compatibility
    n!("check_platform", "🔍 Checking platform compatibility...");
    check_platform_compatibility(&worker)?;
    
    // 3. Download PKGBUILD
    n!("download_pkgbuild", "📄 Downloading PKGBUILD...");
    let pkgbuild_content = download_pkgbuild(&worker_id).await?;
    
    // 4. Parse PKGBUILD
    n!("parse_pkgbuild", "🔍 Parsing PKGBUILD...");
    let pkgbuild = PkgBuild::parse(&pkgbuild_content)?;
    
    // 5. Check dependencies
    n!("check_deps", "🔧 Checking dependencies...");
    check_dependencies(&pkgbuild)?;
    
    // 6. Create temp directories
    let temp_dir = create_temp_directories(&worker_id)?;
    
    // 7. Execute build()
    n!("build_start", "🏗️  Starting build phase...");
    let executor = PkgBuildExecutor::new(
        temp_dir.join("src"),
        temp_dir.join("pkg"),
        temp_dir.clone(),
    );
    
    executor.build(&pkgbuild, |line| {
        n!("build_output", "{}", line);
    }).await?;
    
    // 8. Execute package()
    n!("package_start", "📦 Starting package phase...");
    executor.package(&pkgbuild, |line| {
        n!("package_output", "{}", line);
    }).await?;
    
    // 9. Install binary
    n!("install_binary", "💾 Installing binary...");
    install_binary(&temp_dir, &pkgbuild)?;
    
    // 10. Update capabilities
    n!("update_caps", "📝 Updating capabilities cache...");
    update_capabilities(&worker_id)?;
    
    // 11. Cleanup
    n!("cleanup", "🧹 Cleaning up temp files...");
    cleanup_temp_directories(&temp_dir)?;
    
    n!("install_complete", "✅ Worker installation complete!");
    Ok(())
}
```

## 📋 TODO List

### Priority 1: Backend Implementation

**File:** `bin/20_rbee_hive/src/worker_install.rs`
- [ ] Create module file
- [ ] Implement `handle_worker_install()`
- [ ] Implement `fetch_worker_metadata()` - HTTP GET to catalog
- [ ] Implement `check_platform_compatibility()` - Verify OS/arch
- [ ] Implement `download_pkgbuild()` - HTTP GET PKGBUILD
- [ ] Implement `check_dependencies()` - Verify deps installed
- [ ] Implement `create_temp_directories()` - /tmp/worker-install/
- [ ] Implement `install_binary()` - Copy to /usr/local/bin
- [ ] Implement `update_capabilities()` - Add to capabilities.json
- [ ] Implement `cleanup_temp_directories()` - Remove temp files

**File:** `bin/20_rbee_hive/src/lib.rs`
- [ ] Add `pub mod worker_install;`

**File:** `bin/20_rbee_hive/src/job_router.rs`
- [ ] Add `Operation::WorkerInstall` match arm
- [ ] Call `worker_install::handle_worker_install()`

**File:** `bin/20_rbee_hive/Cargo.toml`
- [ ] Add `reqwest = { version = "0.11", features = ["json"] }`
- [ ] Add `tempfile = "3.8"`

**File:** `bin/97_contracts/operations-contract/src/operation_impl.rs`
- [ ] Add `WorkerInstall` to `name()` method
- [ ] Add `WorkerInstall` to `target_server()` method

### Priority 2: SDK Integration

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs`
- [ ] Add `install_worker()` method
- [ ] Create `WorkerInstall` operation
- [ ] Call `submit_and_stream()`

### Priority 3: React Hooks

**File:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useHiveOperations.ts`
- [ ] Add `installWorker()` method
- [ ] Add `installingWorker` state
- [ ] Add `installError` state
- [ ] Add `installProgress` state

### Priority 4: UI Wiring

**File:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx`
- [ ] Import `useHiveOperations`
- [ ] Call `installWorker(workerId)` in `handleInstall`
- [ ] Show progress from `installProgress`
- [ ] Handle errors from `installError`

**File:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/index.tsx`
- [ ] Query installed workers on mount
- [ ] Update `installedWorkers` state from backend
- [ ] Pass to WorkerCatalogView

### Priority 5: Capabilities Integration

**File:** `bin/20_rbee_hive/ui/app/src/hooks/useWorkerCatalog.ts`
- [ ] Fetch `/v1/capabilities` from hive
- [ ] Filter catalog by capabilities
- [ ] Show only compatible workers

## Data Flow

```
UI Click "Install Worker"
    ↓
useHiveOperations.installWorker(workerId)
    ↓
SDK: HiveOperations.install_worker(workerId)
    ↓
JobClient.submit_and_stream(WorkerInstall { worker_id })
    ↓
POST http://localhost:7835/v1/jobs
    ↓
job_router.rs: route_operation()
    ↓
worker_install::handle_worker_install(worker_id)
    ↓
1. GET http://localhost:8787/workers/{worker_id}
2. GET http://localhost:8787/workers/{worker_id}/PKGBUILD
3. PkgBuild::parse(content)
4. PkgBuildExecutor::build()
5. PkgBuildExecutor::package()
6. Install binary to /usr/local/bin
7. Update capabilities.json
    ↓
SSE Stream: data: ✅ Worker installation complete!
    ↓
UI: Show success message
```

## Key Files

### Backend
- `bin/20_rbee_hive/src/worker_install.rs` - **NEW** - Main handler
- `bin/20_rbee_hive/src/job_router.rs` - Add match arm
- `bin/20_rbee_hive/src/pkgbuild_parser.rs` - ✅ Ready
- `bin/20_rbee_hive/src/pkgbuild_executor.rs` - ✅ Ready

### Contracts
- `bin/97_contracts/operations-contract/src/lib.rs` - ✅ WorkerInstall added
- `bin/97_contracts/operations-contract/src/requests.rs` - ✅ WorkerInstallRequest added
- `bin/97_contracts/operations-contract/src/operation_impl.rs` - Need to update

### Frontend
- `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs` - Add method
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useHiveOperations.ts` - Add hook
- `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx` - ✅ UI ready

### Catalog
- `bin/80-hono-worker-catalog/src/index.ts` - ✅ CORS enabled
- `bin/80-hono-worker-catalog/src/routes.ts` - ✅ Endpoints ready
- `bin/80-hono-worker-catalog/src/data.ts` - ✅ 3 workers ready

## Next Session

**Start with:** Implementing `worker_install.rs` handler

**Key points:**
1. Use `reqwest` for HTTP requests to catalog
2. Use `PkgBuild::parse()` for PKGBUILD parsing
3. Use `PkgBuildExecutor` for build/package execution
4. Use `n!()` macro for SSE streaming
5. Use `tempfile::TempDir` for temp directories
6. Handle all errors gracefully
7. Stream progress to UI in real-time

**Dependencies to add:**
```toml
[dependencies]
reqwest = { version = "0.11", features = ["json"] }
tempfile = "3.8"
```

---

**Status:** Infrastructure complete ✅, Ready for handler implementation 🎯
