# Worker-HTTP Migration Plan

**Date**: 2025-10-05  
**Estimated Time**: 2-3 hours  
**Complexity**: Medium-High  
**Status**: ⏳ Ready to execute

---

## Overview

Migrate the HTTP server layer from `worker-orcd` to `worker-crates/worker-http` to enable code reuse for `worker-aarmd` and other future workers.

## Files to Migrate

### Source Files (7 files)
```
bin/worker-orcd/src/http/
├── execute.rs       (8478 bytes)  - POST /execute endpoint handler
├── health.rs        (1788 bytes)  - GET /health endpoint
├── mod.rs           (649 bytes)   - Module exports
├── routes.rs        (2282 bytes)  - Router setup
├── server.rs        (8139 bytes)  - Axum server lifecycle
├── sse.rs           (9311 bytes)  - Server-Sent Events streaming
└── validation.rs    (27934 bytes) - Request validation logic
```

**Total**: ~58KB of code

---

## Dependencies Analysis

### Current Dependencies in worker-orcd
```toml
axum = "0.7"
tower = { version = "0.5", features = ["util"] }
tokio = { version = "1.0", features = ["full"] }
serde = "1.0"
serde_json = "1.0"
tracing = "0.1"
futures = "0.3"
```

### Cross-Crate Dependencies

**worker-http will depend on**:
- `worker-common` - For `WorkerError`, `SamplingConfig`, `InferenceResult`, `ExecuteRequest`
- `axum` - HTTP framework
- `tower` - Middleware
- `tokio` - Async runtime
- `serde` / `serde_json` - Serialization
- `tracing` - Logging
- `futures` - Async utilities

**worker-http will NOT depend on**:
- `worker-orcd` (CUDA-specific code)
- `worker-tokenizer` (inference layer concern)
- `worker-models` (inference layer concern)

---

## Import Mapping

### Current Imports in HTTP Files

From `grep` analysis:
```rust
// In execute.rs, server.rs, etc.
use crate::error::WorkerError;
use crate::inference_result::InferenceResult;
use crate::sampling_config::SamplingConfig;
use crate::http::validation::ExecuteRequest;
use crate::http::sse::*;
use crate::inference_executor::InferenceExecutor;  // ⚠️ CUDA-specific
```

### Required Import Updates

| Old Import | New Import | Notes |
|------------|------------|-------|
| `use crate::error::WorkerError;` | `use worker_common::WorkerError;` | ✅ Already in worker-common |
| `use crate::inference_result::InferenceResult;` | `use worker_common::InferenceResult;` | ✅ Already in worker-common |
| `use crate::sampling_config::SamplingConfig;` | `use worker_common::SamplingConfig;` | ✅ Already in worker-common |
| `use crate::http::validation::ExecuteRequest;` | `use crate::validation::ExecuteRequest;` | ✅ Stays in worker-http |
| `use crate::http::sse::*;` | `use crate::sse::*;` | ✅ Stays in worker-http |
| `use crate::inference_executor::InferenceExecutor;` | ⚠️ **Keep in worker-orcd** | CUDA-specific |

---

## Migration Strategy

### Phase 1: Preparation (15 min)

1. **Review current HTTP code structure**
   ```bash
   cd bin/worker-orcd/src/http
   wc -l *.rs
   grep -r "use crate::" . | sort | uniq
   ```

2. **Identify CUDA dependencies**
   ```bash
   grep -r "InferenceExecutor\|CudaContext\|CudaError" bin/worker-orcd/src/http/
   ```

3. **Create backup branch**
   ```bash
   git branch migration-worker-http-backup
   ```

### Phase 2: Move Files (20 min)

1. **Remove placeholder**
   ```bash
   rm bin/worker-crates/worker-http/src/lib.rs
   ```

2. **Move HTTP files with git mv**
   ```bash
   git mv bin/worker-orcd/src/http/execute.rs bin/worker-crates/worker-http/src/
   git mv bin/worker-orcd/src/http/health.rs bin/worker-crates/worker-http/src/
   git mv bin/worker-orcd/src/http/mod.rs bin/worker-crates/worker-http/src/lib.rs
   git mv bin/worker-orcd/src/http/routes.rs bin/worker-crates/worker-http/src/
   git mv bin/worker-orcd/src/http/server.rs bin/worker-crates/worker-http/src/
   git mv bin/worker-orcd/src/http/sse.rs bin/worker-crates/worker-http/src/
   git mv bin/worker-orcd/src/http/validation.rs bin/worker-crates/worker-http/src/
   ```

3. **Clean up empty directory**
   ```bash
   rmdir bin/worker-orcd/src/http
   ```

### Phase 3: Update Imports (30 min)

1. **Update imports in worker-http files**
   ```bash
   cd bin/worker-crates/worker-http/src
   
   # Replace crate::error with worker_common
   find . -name '*.rs' -exec sed -i 's/use crate::error::/use worker_common::error::/g' {} \;
   find . -name '*.rs' -exec sed -i 's/use crate::error::WorkerError/use worker_common::WorkerError/g' {} \;
   
   # Replace crate::inference_result with worker_common
   find . -name '*.rs' -exec sed -i 's/use crate::inference_result::/use worker_common::inference_result::/g' {} \;
   find . -name '*.rs' -exec sed -i 's/use crate::InferenceResult/use worker_common::InferenceResult/g' {} \;
   
   # Replace crate::sampling_config with worker_common
   find . -name '*.rs' -exec sed -i 's/use crate::sampling_config::/use worker_common::sampling_config::/g' {} \;
   find . -name '*.rs' -exec sed -i 's/use crate::SamplingConfig/use worker_common::SamplingConfig/g' {} \;
   
   # Replace crate::http:: with crate::
   find . -name '*.rs' -exec sed -i 's/use crate::http::/use crate::/g' {} \;
   ```

2. **Manual review of InferenceExecutor references**
   ```bash
   grep -n "InferenceExecutor" bin/worker-crates/worker-http/src/*.rs
   ```
   
   **Decision**: InferenceExecutor is CUDA-specific and should NOT be in worker-http.
   
   **Solution**: Make worker-http generic over an inference backend trait.

### Phase 4: Refactor for Platform Independence (45 min)

**Problem**: `execute.rs` currently depends on `InferenceExecutor` which is CUDA-specific.

**Solution**: Introduce a trait-based abstraction.

1. **Create inference backend trait in worker-http**
   ```rust
   // bin/worker-crates/worker-http/src/backend.rs
   
   use worker_common::{InferenceResult, SamplingConfig};
   use async_trait::async_trait;
   
   /// Platform-agnostic inference backend
   #[async_trait]
   pub trait InferenceBackend: Send + Sync {
       /// Execute inference with streaming
       async fn execute_streaming(
           &self,
           prompt: &str,
           config: &SamplingConfig,
       ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>>;
       
       /// Cancel inference by job ID
       async fn cancel(&self, job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
   }
   ```

2. **Update execute.rs to use trait**
   ```rust
   // Before
   use crate::inference_executor::InferenceExecutor;
   
   pub async fn execute_handler(
       State(executor): State<Arc<InferenceExecutor>>,
       // ...
   ) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, WorkerError> {
       // ...
   }
   
   // After
   use crate::backend::InferenceBackend;
   
   pub async fn execute_handler<B: InferenceBackend>(
       State(backend): State<Arc<B>>,
       // ...
   ) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, WorkerError> {
       // ...
   }
   ```

3. **Update routes.rs to be generic**
   ```rust
   // Before
   pub fn create_router(executor: Arc<InferenceExecutor>) -> Router {
       // ...
   }
   
   // After
   pub fn create_router<B: InferenceBackend + 'static>(backend: Arc<B>) -> Router {
       // ...
   }
   ```

4. **Update server.rs to be generic**
   ```rust
   // Before
   pub async fn run_server(
       addr: SocketAddr,
       executor: Arc<InferenceExecutor>,
   ) -> Result<(), Box<dyn std::error::Error>> {
       // ...
   }
   
   // After
   pub async fn run_server<B: InferenceBackend + 'static>(
       addr: SocketAddr,
       backend: Arc<B>,
   ) -> Result<(), Box<dyn std::error::Error>> {
       // ...
   }
   ```

### Phase 5: Update Cargo.toml (10 min)

1. **Update worker-http/Cargo.toml**
   ```toml
   [package]
   name = "worker-http"
   version = "0.1.0"
   edition = "2021"
   license = "GPL-3.0-or-later"
   description = "HTTP server and SSE streaming for llama-orch workers"
   
   [dependencies]
   worker-common = { path = "../worker-common" }
   axum = "0.7"
   tower = { version = "0.5", features = ["util"] }
   tokio = { version = "1.0", features = ["full"] }
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   tracing = "0.1"
   futures = "0.3"
   async-trait = "0.1"
   
   [dev-dependencies]
   ```

2. **Update worker-orcd/Cargo.toml**
   ```toml
   [dependencies]
   # ... existing dependencies ...
   worker-http = { path = "../worker-crates/worker-http" }
   ```

3. **Update worker-orcd/src/lib.rs**
   ```rust
   // Remove
   pub mod http;
   
   // worker-orcd will implement InferenceBackend trait
   ```

### Phase 6: Implement InferenceBackend in worker-orcd (20 min)

1. **Create adapter in worker-orcd**
   ```rust
   // bin/worker-orcd/src/http_backend.rs
   
   use worker_http::backend::InferenceBackend;
   use worker_common::{InferenceResult, SamplingConfig};
   use crate::inference_executor::InferenceExecutor;
   use async_trait::async_trait;
   
   pub struct CudaInferenceBackend {
       executor: Arc<InferenceExecutor>,
   }
   
   impl CudaInferenceBackend {
       pub fn new(executor: Arc<InferenceExecutor>) -> Self {
           Self { executor }
       }
   }
   
   #[async_trait]
   impl InferenceBackend for CudaInferenceBackend {
       async fn execute_streaming(
           &self,
           prompt: &str,
           config: &SamplingConfig,
       ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
           self.executor.execute_streaming(prompt, config).await
       }
       
       async fn cancel(&self, job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
           self.executor.cancel(job_id).await
       }
   }
   ```

2. **Update worker-orcd/src/main.rs**
   ```rust
   use worker_http::server::run_server;
   use crate::http_backend::CudaInferenceBackend;
   
   #[tokio::main]
   async fn main() -> Result<(), Box<dyn std::error::Error>> {
       // ... setup code ...
       
       let executor = Arc::new(InferenceExecutor::new(/* ... */));
       let backend = Arc::new(CudaInferenceBackend::new(executor));
       
       run_server(addr, backend).await?;
       
       Ok(())
   }
   ```

### Phase 7: Verification (20 min)

1. **Compile worker-http**
   ```bash
   cargo check -p worker-http
   ```

2. **Compile worker-orcd**
   ```bash
   cargo check -p worker-orcd
   ```

3. **Run tests**
   ```bash
   cargo test -p worker-http --lib
   cargo test -p worker-orcd
   ```

4. **Verify git history**
   ```bash
   git log --follow bin/worker-crates/worker-http/src/server.rs
   ```

### Phase 8: Commit (5 min)

```bash
git add -A
git commit -m "refactor: extract worker-http from worker-orcd

- Move src/http/ to worker-crates/worker-http
- Introduce InferenceBackend trait for platform independence
- Implement CudaInferenceBackend in worker-orcd
- Update imports to use worker-common
- All tests passing

Refs: .docs/WORKER_AARMD_DEVELOPMENT_PLAN.md Phase 1.5"
```

---

## Challenges & Solutions

### Challenge 1: CUDA-Specific Dependencies

**Problem**: `execute.rs` directly uses `InferenceExecutor` which is CUDA-specific.

**Solution**: Introduce `InferenceBackend` trait to abstract platform-specific inference.

**Benefits**:
- ✅ worker-http becomes platform-agnostic
- ✅ worker-aarmd can implement same trait for Metal
- ✅ Easier to test with mock backends

### Challenge 2: SSE Streaming Complexity

**Problem**: SSE streaming logic is tightly coupled with CUDA inference.

**Solution**: Keep SSE logic in worker-http, but make it generic over the backend trait.

**Implementation**:
```rust
// worker-http handles SSE framing
// Backend trait provides token stream
pub async fn execute_handler<B: InferenceBackend>(
    backend: Arc<B>,
    // ...
) -> Sse<impl Stream> {
    let result = backend.execute_streaming(prompt, config).await?;
    
    // worker-http converts result to SSE events
    let stream = stream::iter(result.tokens)
        .map(|token| Event::default().data(token));
    
    Sse::new(stream)
}
```

### Challenge 3: Validation Logic Size

**Problem**: `validation.rs` is 27KB - largest file in HTTP layer.

**Solution**: Move as-is initially, refactor later if needed.

**Note**: Validation logic is pure Rust and has no CUDA dependencies.

---

## Testing Strategy

### Unit Tests

**In worker-http**:
- ✅ Request validation logic (already in `validation.rs`)
- ✅ SSE event formatting
- ✅ Error response serialization

**In worker-orcd**:
- ✅ CudaInferenceBackend implementation
- ✅ Integration with InferenceExecutor

### Integration Tests

**Mock Backend for Testing**:
```rust
// bin/worker-crates/worker-http/tests/mock_backend.rs

use worker_http::backend::InferenceBackend;
use async_trait::async_trait;

pub struct MockBackend {
    responses: Vec<InferenceResult>,
}

#[async_trait]
impl InferenceBackend for MockBackend {
    async fn execute_streaming(/* ... */) -> Result<InferenceResult, _> {
        Ok(self.responses[0].clone())
    }
    
    async fn cancel(&self, _job_id: &str) -> Result<(), _> {
        Ok(())
    }
}

#[tokio::test]
async fn test_execute_endpoint_with_mock() {
    let backend = Arc::new(MockBackend::new(/* ... */));
    let app = worker_http::routes::create_router(backend);
    
    // Test HTTP endpoint with mock backend
    // ...
}
```

---

## Rollback Plan

If migration fails:

```bash
# Reset to backup branch
git reset --hard migration-worker-http-backup

# Or revert specific commit
git revert <commit-hash>
```

---

## Success Criteria

- [ ] All 7 HTTP files moved to worker-http
- [ ] worker-http compiles without errors
- [ ] worker-orcd compiles without errors
- [ ] InferenceBackend trait defined
- [ ] CudaInferenceBackend implemented in worker-orcd
- [ ] All imports updated correctly
- [ ] Git history preserved for all files
- [ ] Tests pass in both crates
- [ ] Documentation updated

---

## Timeline Breakdown

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Preparation | 15 min |
| 2 | Move files | 20 min |
| 3 | Update imports | 30 min |
| 4 | Refactor for platform independence | 45 min |
| 5 | Update Cargo.toml | 10 min |
| 6 | Implement InferenceBackend in worker-orcd | 20 min |
| 7 | Verification | 20 min |
| 8 | Commit | 5 min |
| **Total** | | **~2h 45min** |

---

## Post-Migration Benefits

### For worker-aarmd
- ✅ Reuse entire HTTP server layer
- ✅ Only implement InferenceBackend trait for Metal
- ✅ Identical API surface to worker-orcd

### For Testing
- ✅ Mock backends for HTTP endpoint testing
- ✅ No CUDA required for HTTP layer tests
- ✅ Faster CI/CD pipeline

### For Maintenance
- ✅ HTTP bugs fixed once, benefit all workers
- ✅ API changes centralized
- ✅ Clear separation of concerns

---

## References

- **worker-http stub**: `bin/worker-crates/worker-http/`
- **Current HTTP code**: `bin/worker-orcd/src/http/`
- **worker-common**: `bin/worker-crates/worker-common/`
- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **Migration Complete**: `.docs/WORKER_CRATES_MIGRATION_COMPLETE.md`

---

**Status**: ⏳ Ready to execute  
**Estimated Time**: 2-3 hours  
**Complexity**: Medium-High (requires trait abstraction)  
**Blocker**: None (all dependencies migrated)
