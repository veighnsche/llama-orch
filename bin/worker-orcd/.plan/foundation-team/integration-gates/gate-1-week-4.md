# Gate 1: Foundation Complete - Week 4

**Date**: [TBD - End of Week 4]  
**Status**: 🟡 Not Started  
**Critical**: YES - Blocks Teams 2 & 3 if failed

---

## Gate Criteria

### Foundation Team Deliverables

#### HTTP Server
- [ ] Server starts on specified port
- [ ] All endpoints respond correctly:
  - [ ] POST /execute (accepts requests, returns 202)
  - [ ] GET /health (returns status)
  - [ ] POST /cancel (accepts job_id, returns 202)
- [ ] Server handles SIGTERM gracefully
- [ ] Correlation ID middleware working

#### SSE Streaming
- [ ] SSE stream established (`Content-Type: text/event-stream`)
- [ ] Events emitted in correct order: started → token* → end/error
- [ ] UTF-8 boundary buffering implemented
- [ ] No invalid UTF-8 sequences emitted
- [ ] Backpressure handling for slow clients

#### FFI Layer
- [ ] C API interface defined and documented (`cuda_api.h`)
- [ ] Rust FFI bindings implemented (`cuda_ffi.rs`)
- [ ] Error codes propagate correctly (C++ → Rust → HTTP)
- [ ] Memory safety verified (no leaks, valgrind clean)
- [ ] Integration test: Rust calls C++ function successfully

#### CUDA Context
- [ ] CUDA context initializes successfully
- [ ] Device selection working (--gpu-device flag)
- [ ] VRAM-only mode enforced (UMA disabled)
- [ ] Device properties readable
- [ ] Context cleanup on shutdown (no resource leaks)

#### Shared Kernels
- [ ] Embedding lookup kernel implemented and tested
- [ ] cuBLAS GEMM wrapper working
- [ ] Temperature scaling kernel working
- [ ] Greedy sampling (temp=0) working
- [ ] Stochastic sampling (temp>0) working
- [ ] Seeded RNG produces reproducible results

#### KV Cache
- [ ] KV cache allocation working
- [ ] Cache size calculated correctly
- [ ] Cache initialized to zero
- [ ] Cache freed on inference completion
- [ ] OOM handling if cache allocation fails

#### Integration Tests
- [ ] Test framework set up (Rust integration tests)
- [ ] Test: HTTP → FFI → CUDA context init
- [ ] Test: HTTP → Execute → SSE stream
- [ ] Test: Sampling reproducibility (same seed → same output)
- [ ] Test: Error propagation (CUDA error → HTTP 500)

---

## Gate Test

### Automated Test Suite

**Command**:
```bash
cd bin/worker-orcd
cargo test --features cuda -- --test-threads=1
```

**Expected Output**:
```
running 15 tests
test http_server_starts ... ok
test execute_endpoint_validation ... ok
test sse_streaming_order ... ok
test ffi_context_init ... ok
test cuda_context_creates ... ok
test embedding_kernel ... ok
test gemm_wrapper ... ok
test sampling_greedy ... ok
test sampling_stochastic ... ok
test kv_cache_allocation ... ok
test integration_http_to_cuda ... ok
test integration_sse_stream ... ok
test error_propagation ... ok
test utf8_boundary_safety ... ok
test reproducibility_same_seed ... ok

test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Manual Smoke Test

**Test 1: Server Starts**
```bash
./target/release/worker-orcd --port 8080 --gpu-device 0 --model dummy.gguf
# Expected: Server starts, logs "Worker starting on port 8080"
```

**Test 2: Health Check**
```bash
curl http://localhost:8080/health
# Expected: {"status":"starting","model":null,"resident":false,...}
```

**Test 3: Execute Request (Mock)**
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{"job_id":"test","prompt":"Hello","max_tokens":10,"temperature":0.0,"seed":42}'
# Expected: 202 Accepted (no actual inference yet, but request accepted)
```

**Test 4: FFI Call**
```bash
# Run integration test that calls CUDA context init via FFI
cargo test --features cuda ffi_context_init
# Expected: Test passes, context created and destroyed
```

---

## Results

### Test Run 1

**Date**: [TBD]  
**Outcome**: ⏸️ Not Run  
**Notes**: [To be filled during Week 4]

**Failures**: [List any failures]

**Action Items**:
- [ ] Fix X (Owner: [Name])
- [ ] Fix Y (Owner: [Name])

---

## Go/No-Go Decision

**Decision**: ⏸️ Pending  
**Decision Date**: [End of Week 4]  
**Decision Maker**: [Team Lead + PM]

### Go Criteria
- [ ] All automated tests passing
- [ ] Manual smoke tests passing
- [ ] No critical bugs
- [ ] Teams 2 & 3 can start their work

### No-Go Criteria
- ❌ Any automated test failing
- ❌ FFI interface unstable
- ❌ CUDA context not working
- ❌ Critical bugs in HTTP/SSE layer

### If No-Go

**Impact**: Teams 2 & 3 blocked, Week 5 delayed

**Action Plan**:
1. Emergency debug session (all hands)
2. Identify root cause
3. Fix critical issues
4. Re-run gate test
5. Delay Week 5 start if needed

---

## Handoff to Teams 2 & 3

### What's Ready

**For Llama Team**:
- [ ] FFI interface documented and stable
- [ ] Shared kernels (embedding, GEMM, sampling) available
- [ ] Integration test examples to follow
- [ ] `interfaces.md` document complete

**For GPT Team**:
- [ ] Same as Llama Team
- [ ] HTTP API stable for testing
- [ ] SSE streaming working for validation

### What's Not Ready (Expected)

- ❌ Model loading (Teams 2 & 3 responsibility)
- ❌ Tokenization (Teams 2 & 3 responsibility)
- ❌ Architecture-specific kernels (Teams 2 & 3 responsibility)

---

## Metrics

### Test Coverage
- **Target**: >80% coverage for new code
- **Actual**: [To be measured]

### Performance Baseline
- **CUDA context init**: [To be measured] ms
- **HTTP request latency**: [To be measured] ms
- **SSE event latency**: [To be measured] ms

### Defect Count
- **Critical**: [Count]
- **Major**: [Count]
- **Minor**: [Count]

---

## Lessons Learned

### What Went Well
- [To be filled during retrospective]

### What Didn't Go Well
- [To be filled during retrospective]

### Action Items for Next Sprint
- [To be filled during retrospective]

---

**Status**: 🟡 In Progress (Week 4)  
**Next Review**: End of Week 4 (Friday demo)  
**Blocker**: None currently
