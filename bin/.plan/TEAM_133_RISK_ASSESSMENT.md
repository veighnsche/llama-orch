# TEAM-133: Risk Assessment & Mitigation

**Detailed risk analysis for decomposition of llm-worker-rbee**

---

## OVERALL RISK LEVEL: ⚠️ MEDIUM-HIGH

**Justification:**
- ✅ **Pilot success:** worker-rbee-error already done (TEAM-130)
- ✅ **Clean architecture:** Well-separated concerns
- ⚠️ **Complex modules:** HTTP server (1,280 LOC) and inference (1,300 LOC)
- ⚠️ **Integration risk:** Tight coupling with rbee-hive and queen-rbee

---

## RISK BREAKDOWN BY CRATE

### 1. worker-rbee-error (✅ LOW RISK)

**Status:** ✅ ALREADY COMPLETE (TEAM-130)

**Risk Level:** **NONE** - Already migrated and tested!

**Evidence:**
- 100% test coverage (TEAM-130)
- All tests passing
- Clean separation from rest of codebase
- No breaking changes

**Mitigation:** N/A - Already complete

---

### 2. worker-rbee-startup (⚠️ MEDIUM RISK)

**Risk Level:** ⚠️ **MEDIUM**

**Risk Factors:**
1. **Integration with rbee-hive** (callback protocol)
   - Risk: Breaking callback format breaks worker registration
   - Probability: Low (well-tested by TEAM-130)
   - Impact: HIGH (workers won't register!)

2. **Network errors** (HTTP client)
   - Risk: Retry logic not implemented
   - Probability: Medium
   - Impact: Medium (temporary failures)

3. **Serialization changes**
   - Risk: JSON format changes break compatibility
   - Probability: Low
   - Impact: HIGH

**Mitigation Strategies:**
1. ✅ **Keep callback format stable**
   - Define contract test with rbee-hive
   - Version callback API (`/v1/ready`)

2. ✅ **Add retry logic**
   - Exponential backoff
   - Max 3 retries
   - Log all attempts

3. ✅ **Contract tests**
   - Mock rbee-hive responses
   - Test all error scenarios
   - Verify JSON serialization

**Test Requirements:**
- [x] Callback success (TEAM-130 done)
- [x] Callback failure (TEAM-130 done)
- [x] Network error (TEAM-130 done)
- [ ] Retry logic (TODO)
- [ ] Contract test with rbee-hive (TODO)

**Timeline Impact:** +1 day (add retry logic)

---

### 3. worker-rbee-health (✅ LOW RISK)

**Risk Level:** ✅ **LOW**

**Risk Factors:**
1. **Background task management**
   - Risk: Task panics or leaks
   - Probability: Low (simple loop)
   - Impact: Medium (heartbeats stop)

2. **HTTP client errors**
   - Risk: Heartbeat failures accumulate
   - Probability: Medium (network issues)
   - Impact: LOW (non-fatal by design)

**Mitigation Strategies:**
1. ✅ **Panic handling**
   - Wrap task in catch_unwind (already done)
   - Log panics but don't crash worker

2. ✅ **Error tolerance**
   - Already non-fatal by design
   - Logs errors but continues

**Test Requirements:**
- [x] Heartbeat success (TEAM-115 done)
- [x] Heartbeat failure handling (TEAM-115 done)
- [x] Config builder (TEAM-115 done)
- [ ] Task lifecycle (TODO - spawns and stops)

**Timeline Impact:** None

---

### 4. worker-rbee-sse-streaming (⚠️ MEDIUM RISK)

**Risk Level:** ⚠️ **MEDIUM**

**Risk Factors:**
1. **Event format breaking changes**
   - Risk: Clients (queen-rbee, rbee-keeper) expect old format
   - Probability: HIGH (refactoring to generics)
   - Impact: HIGH (all inference breaks!)

2. **Generic refactoring complexity**
   - Risk: Generics make API harder to use
   - Probability: Medium
   - Impact: Medium (developer friction)

3. **Serialization changes**
   - Risk: JSON output changes break clients
   - Probability: High (generics change JSON structure)
   - Impact: HIGH

**Mitigation Strategies:**
1. 🔴 **CRITICAL: Keep JSON format backward compatible**
   ```rust
   // BEFORE (current)
   { "type": "token", "t": "hello", "i": 0 }
   
   // AFTER (generic) - MUST produce same JSON!
   { "type": "token", "t": "hello", "i": 0 }
   ```

2. ✅ **Type aliases for compatibility**
   ```rust
   pub type LlmEvent = InferenceEvent<TokenOutput>;
   // Clients use LlmEvent, not generic version
   ```

3. ✅ **Contract tests**
   - Test JSON serialization for all event types
   - Compare with golden files
   - Catch breaking changes early

**Test Requirements:**
- [ ] Event serialization (all types)
- [ ] Generic type instantiation
- [ ] Backward compatibility (golden files)
- [ ] Integration with queen-rbee (TODO)
- [ ] Integration with rbee-keeper (TODO)

**Timeline Impact:** +2 days (generics + contract tests)

---

### 5. worker-rbee-http-server (🔴 HIGH RISK)

**Risk Level:** 🔴 **HIGH**

**Risk Factors:**
1. **Largest module** (1,280 LOC across 10 files)
   - Risk: Complex refactoring, easy to break
   - Probability: High
   - Impact: HIGH (entire HTTP API breaks)

2. **Route configuration changes**
   - Risk: Route paths change, clients break
   - Probability: Low (just moving code)
   - Impact: HIGH

3. **Middleware ordering**
   - Risk: Auth/correlation middleware order matters
   - Probability: Medium
   - Impact: Medium (auth failures)

4. **Trait design** (InferenceBackend)
   - Risk: Trait too LLM-specific, hard to reuse
   - Probability: Medium
   - Impact: HIGH (blocks future workers)

5. **Validation module** (691 LOC!)
   - Risk: Should use input-validation, but duplicated logic
   - Probability: Low (can refactor later)
   - Impact: Medium (maintenance burden)

**Mitigation Strategies:**
1. 🔴 **CRITICAL: Keep route paths stable**
   - Document all endpoints in contract
   - No route changes during migration
   - Version all routes (`/v1/...`)

2. ✅ **Test middleware ordering**
   ```rust
   #[tokio::test]
   async fn test_middleware_order() {
       // 1. Correlation ID added first
       // 2. Auth check second
       // 3. Handler runs last
   }
   ```

3. ✅ **Generic trait design**
   ```rust
   // Current - good!
   pub trait InferenceBackend: Send + Sync {
       async fn execute(&mut self, req: ExecuteRequest) -> Result<InferenceResult>;
       // ...
   }
   // No LLM-specific methods!
   ```

4. 🟡 **Validation refactoring** (DEFER)
   - Keep manual validation for now
   - Refactor to input-validation in Phase 2
   - Non-blocking issue

**Test Requirements:**
- [x] Server lifecycle (done)
- [x] Route configuration (done)
- [x] Auth middleware (TEAM-102 done)
- [ ] Correlation middleware integration
- [ ] All endpoints (health, ready, execute, loading)
- [ ] Error responses (all error codes)
- [ ] SSE streaming (backpressure, connection close)
- [ ] Integration test with mock backend

**Timeline Impact:** +3 days (comprehensive endpoint testing)

---

### 6. worker-rbee-inference-base (🔴 VERY HIGH RISK)

**Risk Level:** 🔴 **VERY HIGH**

**Risk Factors:**
1. **Most complex module** (1,300 LOC, 12 files)
   - Risk: Hard to test, easy to break
   - Probability: HIGH
   - Impact: CRITICAL (inference breaks!)

2. **Candle integration**
   - Risk: Device management, tensor operations fragile
   - Probability: Medium
   - Impact: CRITICAL

3. **Model loading** (SafeTensors, GGUF)
   - Risk: Format parsing errors, memory issues
   - Probability: Medium
   - Impact: CRITICAL (models won't load)

4. **Tokenizer loading**
   - Risk: Multiple formats (HF, GGUF-embedded)
   - Probability: Medium
   - Impact: HIGH (tokenization fails)

5. **Inference loop** (autoregressive generation)
   - Risk: Off-by-one errors, cache corruption
   - Probability: Medium
   - Impact: CRITICAL (wrong outputs!)

6. **Sampling logic**
   - Risk: Probability distributions, numerical stability
   - Probability: Low (well-tested)
   - Impact: Medium (output quality)

7. **LLM-specific bias**
   - Risk: Hard to reuse for non-LLM workers
   - Probability: HIGH
   - Impact: HIGH (blocks future workers)

**Mitigation Strategies:**
1. 🔴 **CRITICAL: Comprehensive inference tests**
   - Test all model architectures (Llama, Mistral, Phi, Qwen)
   - Test both SafeTensors and GGUF formats
   - Compare outputs with known-good results
   - Test edge cases (empty prompt, max_tokens=1, etc.)

2. 🔴 **CRITICAL: Device management tests**
   ```rust
   #[test]
   fn test_cpu_device() { ... }
   
   #[test]
   #[cfg(feature = "cuda")]
   fn test_cuda_device() { ... }
   
   #[test]
   #[cfg(target_os = "macos")]
   fn test_metal_device() { ... }
   ```

3. ✅ **Model loading tests**
   - Use small test models (TinyLlama)
   - Test SafeTensors and GGUF
   - Test error handling (corrupt files, OOM)

4. ✅ **Split LLM-specific code**
   ```
   worker-rbee-inference-base/    (generic)
   ├── device.rs                  (generic)
   ├── model_loader.rs            (generic)
   └── vram_tracker.rs            (generic)
   
   llm-worker-rbee-inference/     (LLM-specific)
   ├── tokenizer.rs               (LLM-specific)
   ├── generation.rs              (LLM-specific)
   ├── sampling.rs                (LLM-specific)
   └── models/                    (LLM-specific)
   ```

5. 🟡 **Performance regression tests**
   - Benchmark inference speed before/after
   - Acceptable: <5% slowdown
   - Red flag: >10% slowdown

**Test Requirements:**
- [x] Multi-model support (TEAM-017 done)
- [x] GGUF tokenizer (TEAM-090 done)
- [ ] SafeTensors loading (all architectures)
- [ ] GGUF loading (all architectures)
- [ ] Tokenizer loading (HF + GGUF)
- [ ] Device initialization (CPU/CUDA/Metal)
- [ ] Inference loop correctness
- [ ] Sampling correctness
- [ ] Cache management (warmup, reset)
- [ ] Memory leak detection
- [ ] Performance benchmarks

**Timeline Impact:** +5 days (comprehensive testing + refactoring)

---

## INTEGRATION RISKS

### rbee-hive Integration

**Risk:** Worker registration breaks

**Endpoints affected:**
- `POST <callback_url>/ready` (startup)
- `POST <callback_url>/v1/heartbeat` (health)

**Mitigation:**
1. ✅ Contract tests with rbee-hive
2. ✅ Version all endpoints
3. ✅ Keep JSON formats stable

**Test plan:**
- [ ] Mock rbee-hive server
- [ ] Test all callback scenarios
- [ ] Test heartbeat scenarios
- [ ] Test error responses

---

### queen-rbee Integration

**Risk:** Inference API breaks

**Endpoints affected:**
- `GET http://worker:8080/health` (health check)
- `POST http://worker:8080/v1/inference` (inference SSE)

**Mitigation:**
1. ✅ Keep route paths stable
2. ✅ Keep SSE event format stable
3. ✅ Version all endpoints

**Test plan:**
- [ ] Mock queen-rbee requests
- [ ] Test SSE streaming
- [ ] Test error handling
- [ ] Test cancellation

---

### rbee-keeper Integration

**Risk:** CLI breaks (reads SSE events)

**Mitigation:**
1. ✅ Keep SSE event format stable
2. ✅ Test JSON serialization

**Test plan:**
- [ ] Parse all SSE event types
- [ ] Verify JSON format matches
- [ ] Test cute/story fields

---

## MIGRATION RISKS

### Code Movement Risk

**Risk:** Files moved to wrong crates

**Mitigation:**
1. ✅ Use this investigation report as guide
2. ✅ Move one file at a time
3. ✅ Run tests after each file
4. ✅ Use git to track moves

### Import Path Changes

**Risk:** Broken imports after migration

**Mitigation:**
1. ✅ Use cargo check after each file
2. ✅ Fix imports immediately
3. ✅ Use IDE refactoring tools

### Test Breakage

**Risk:** Tests fail after migration

**Mitigation:**
1. ✅ Run full test suite after each crate
2. ✅ Fix failing tests before moving to next crate
3. ✅ Add new tests for crate boundaries

---

## ROLLBACK PLAN

### Per-Crate Rollback

**If a crate migration fails:**
1. ✅ Git revert the migration commits
2. ✅ Re-run tests to verify rollback
3. ✅ Document failure reason
4. ✅ Fix issue before retrying

### Full Rollback

**If entire migration fails:**
1. ✅ Git revert all migration commits
2. ✅ Return to monolithic binary
3. ✅ Re-evaluate approach
4. ✅ Get additional help from TEAM-130

---

## GO/NO-GO CRITERIA

### Per-Crate Go/No-Go

**Criteria for moving to next crate:**
- [ ] All tests passing
- [ ] No clippy warnings
- [ ] Cargo check passes
- [ ] Integration tests pass
- [ ] Performance acceptable (<5% regression)
- [ ] Peer review complete

### Overall Go/No-Go

**Criteria for declaring success:**
- [ ] All 6 crates created
- [ ] Binary still works
- [ ] All tests passing
- [ ] Integration tests pass
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Peer review complete

---

## TIMELINE IMPACT SUMMARY

| Crate | Base Effort | Risk Mitigation | Total |
|-------|-------------|-----------------|-------|
| 1. worker-rbee-error | 0 days | 0 days | **0 days** ✅ |
| 2. worker-rbee-startup | 1 day | +1 day | **2 days** |
| 3. worker-rbee-health | 1 day | 0 days | **1 day** |
| 4. worker-rbee-sse-streaming | 2 days | +2 days | **4 days** |
| 5. worker-rbee-http-server | 3 days | +3 days | **6 days** |
| 6. worker-rbee-inference-base | 4 days | +5 days | **9 days** |
| **TOTAL** | **11 days** | **+11 days** | **22 days** |

**Timeline:** ~4.5 weeks (with risk mitigation)

**Recommendation:** Allocate 5 weeks (1 week buffer for unknowns)

---

## FINAL RECOMMENDATION

**GO** - Proceed with decomposition

**Confidence Level:** 75% (High confidence in success with proper risk mitigation)

**Critical Success Factors:**
1. 🔴 Start with low-risk crates (error, health, startup)
2. 🔴 Keep all JSON formats backward compatible
3. 🔴 Comprehensive testing at each step
4. 🔴 Peer review after each crate
5. 🔴 Performance benchmarks to catch regressions
