# BDD Feature Files - Gap Analysis
**Date:** 2025-10-11  
**Analyzed by:** TEAM-079  
**Status:** 🔍 Comprehensive Review

---

## Executive Summary

**Total Feature Files:** 16  
**Scenarios Analyzed:** ~80+  
**Coverage:** Good foundation, but **significant gaps** in critical areas

### Critical Gaps Found:
1. ❌ **No concurrent worker scenarios** (race conditions, slot contention)
2. ❌ **No model catalog edge cases** (corruption, migration, schema evolution)
3. ❌ **No registry failure scenarios** (split-brain, network partitions)
4. ❌ **No performance/load testing scenarios**
5. ❌ **No graceful degradation scenarios**
6. ⚠️ **Limited multi-node orchestration scenarios**
7. ⚠️ **Minimal security/auth testing**

---

## 1. Model Catalog (020-model-catalog.feature)

### ✅ What's Covered:
- Basic CRUD: find, insert, query by provider
- Model not found handling
- File size calculation
- Registration after download

### ❌ Critical Gaps:

#### 1.1 Concurrent Access
```gherkin
Scenario: Concurrent model registration from multiple rbee-hive instances
  Given 3 rbee-hive instances are downloading the same model
  When all 3 attempt to register simultaneously
  Then only one INSERT succeeds
  And the other 2 detect duplicate and skip
  And no database locks occur
```

#### 1.2 Catalog Corruption
```gherkin
Scenario: Corrupted catalog database recovery
  Given the SQLite database is corrupted
  When rbee-hive attempts to query catalog
  Then rbee-hive detects corruption
  And rbee-hive creates backup of corrupted DB
  And rbee-hive initializes fresh catalog
  And rbee-keeper displays recovery steps
```

#### 1.3 Schema Migration
```gherkin
Scenario: Catalog schema version mismatch
  Given catalog is v1 schema (no vram_required column)
  When rbee-hive v2 attempts to query
  Then rbee-hive detects schema mismatch
  And rbee-hive runs migration to v2
  And existing data is preserved
```

#### 1.4 Orphaned Entries
```gherkin
Scenario: Cleanup orphaned catalog entries
  Given catalog contains model "/models/deleted.gguf"
  And the file no longer exists on disk
  When rbee-hive runs catalog cleanup
  Then orphaned entry is removed
  And rbee-keeper displays cleanup summary
```

#### 1.5 Large Catalog Performance
```gherkin
Scenario: Query performance with 1000+ models
  Given catalog contains 1000 model entries
  When rbee-hive queries by provider
  Then query completes in <100ms
  And results are paginated
```

---

## 2. queen-rbee Registry (050-queen-rbee-worker-registry.feature)

### ✅ What's Covered:
- Basic CRUD: register, list, filter, update, remove
- Stale worker cleanup
- Capability filtering

### ❌ Critical Gaps:

#### 2.1 Split-Brain Scenarios
```gherkin
Scenario: Network partition causes split-brain registry
  Given queen-rbee-1 has workers [A, B]
  And queen-rbee-2 has workers [C, D]
  When network partition occurs
  And both instances receive requests
  Then each instance maintains separate state
  And when partition heals, conflict resolution occurs
  And duplicate workers are deduplicated
```

#### 2.2 Worker Heartbeat Failures
```gherkin
Scenario: Worker heartbeat timeout with active requests
  Given worker-001 is processing request
  When heartbeat times out (>120s)
  Then queen-rbee marks worker as "stale-but-busy"
  And new requests are NOT routed to worker-001
  And existing request is allowed to complete
  And worker is removed after request completes or 5min timeout
```

#### 2.3 Registry Overflow
```gherkin
Scenario: Registry reaches maximum worker limit
  Given queen-rbee has 1000 workers registered (max limit)
  When rbee-hive attempts to register worker-1001
  Then queen-rbee rejects registration
  And returns error "REGISTRY_FULL"
  And suggests removing stale workers
```

#### 2.4 Concurrent State Updates
```gherkin
Scenario: Race condition on worker state update
  Given worker-001 state is "idle"
  When request-A updates state to "busy" at T+0ms
  And request-B updates state to "busy" at T+1ms
  Then only one update succeeds
  And the other receives "WORKER_ALREADY_BUSY"
  And no state corruption occurs
```

#### 2.5 Worker Capability Changes
```gherkin
Scenario: Worker capabilities change after registration
  Given worker-001 registered with ["cuda:0"]
  When GPU driver crashes
  And worker re-registers with ["cpu"]
  Then queen-rbee updates capabilities
  And in-flight requests to cuda:0 are failed
  And new requests use cpu capability
```

---

## 3. Model Provisioner (030-model-provisioner.feature)

### ✅ What's Covered:
- Download with progress tracking
- Error handling: 404, 403, timeout, checksum
- GGUF support
- Retry logic

### ❌ Critical Gaps:

#### 3.1 Concurrent Downloads
```gherkin
Scenario: Multiple rbee-hive instances download same model
  Given 3 rbee-hive instances need tinyllama-q4
  When all 3 start download simultaneously
  Then only one downloads
  And the other 2 wait for completion
  And all 3 register in catalog after download completes
```

#### 3.2 Partial Download Resume
```gherkin
Scenario: Resume interrupted download from checkpoint
  Given model download interrupted at 60% (3MB/5MB)
  When rbee-hive restarts download
  Then rbee-hive sends Range: bytes=3145728- header
  And download resumes from 60%
  And progress shows "Resuming from 60%..."
```

#### 3.3 Download Bandwidth Limiting
```gherkin
Scenario: Bandwidth throttling for background downloads
  Given system bandwidth limit is 10 MB/s
  When model download starts
  Then download speed is throttled to 10 MB/s
  And other network traffic is not starved
```

#### 3.4 Model Verification Beyond Checksum
```gherkin
Scenario: GGUF file structure validation
  Given model download completes
  When rbee-hive validates GGUF structure
  And file header is corrupted
  Then rbee-hive detects invalid GGUF magic number
  And file is deleted
  And download is retried
```

#### 3.5 Hugging Face Rate Limiting
```gherkin
Scenario: Hugging Face API rate limit exceeded
  Given HF API returns 429 Too Many Requests
  When rbee-hive attempts download
  Then rbee-hive reads Retry-After header
  And waits specified duration
  And retries download
```

---

## 4. Worker Provisioning (040-worker-provisioning.feature)

### ✅ What's Covered:
- Basic worker build and startup
- Feature flag validation
- Binary verification

### ❌ Critical Gaps:

#### 4.1 Build Cache Management
```gherkin
Scenario: Incremental builds with cargo cache
  Given worker binary was built 1 hour ago
  When rbee-hive rebuilds with same features
  Then cargo uses cached artifacts
  And build completes in <10s (vs 2min cold build)
```

#### 4.2 Build Failure Recovery
```gherkin
Scenario: Worker build fails mid-compilation
  Given cargo build starts
  When compilation fails with linker error
  Then rbee-hive captures full error output
  And rbee-keeper displays actionable error
  And suggests checking CUDA toolkit installation
```

#### 4.3 Feature Flag Conflicts
```gherkin
Scenario: Incompatible feature flags detected
  Given user requests --features "cuda,metal"
  When rbee-hive validates features
  Then validation fails
  And error "INCOMPATIBLE_FEATURES" is returned
  And message explains cuda and metal are mutually exclusive
```

#### 4.4 Cross-Compilation
```gherkin
Scenario: Build worker for different architecture
  Given rbee-keeper runs on x86_64
  And target node is aarch64
  When rbee-hive builds worker
  Then cargo cross-compiles for aarch64
  And binary is verified on target architecture
```

---

## 5. Worker Resource Preflight (090-worker-resource-preflight.feature)

### ✅ What's Covered:
- RAM checks
- VRAM checks
- Backend availability
- Disk space checks

### ❌ Critical Gaps:

#### 5.1 Dynamic Resource Monitoring
```gherkin
Scenario: Available RAM decreases during preflight
  Given preflight starts with 8GB available
  When another process allocates 4GB during check
  Then preflight re-checks RAM
  And fails if now insufficient
```

#### 5.2 Multi-GPU Selection
```gherkin
Scenario: Select least-loaded GPU automatically
  Given device 0 has 2GB VRAM free
  And device 1 has 6GB VRAM free
  When user requests --backend cuda (no device specified)
  Then rbee-hive selects device 1 (most free VRAM)
  And logs selection reasoning
```

#### 5.3 CPU Affinity
```gherkin
Scenario: CPU backend with core pinning
  Given system has 16 CPU cores
  When rbee-hive starts CPU worker
  Then worker is pinned to cores 8-15
  And other cores remain available for system
```

#### 5.4 Temperature Monitoring
```gherkin
Scenario: GPU temperature exceeds safe threshold
  Given GPU temperature is 85°C (threshold: 80°C)
  When rbee-hive checks GPU health
  Then preflight fails with "GPU_TOO_HOT"
  And suggests waiting for cooling
```

---

## 6. Input Validation (140-input-validation.feature)

### ✅ What's Covered:
- Model reference format
- Backend name validation
- Device number validation
- API key validation

### ❌ Critical Gaps:

#### 6.1 Prompt Injection Protection
```gherkin
Scenario: Detect potential prompt injection
  Given user provides prompt with system instructions
  When rbee-keeper validates prompt
  Then suspicious patterns are detected
  And warning is displayed
  And user must confirm to proceed
```

#### 6.2 Resource Limit Validation
```gherkin
Scenario: Max tokens exceeds model context window
  Given model context window is 2048 tokens
  When user requests --max-tokens 4096
  Then validation fails
  And error explains model limit
  And suggests reducing max_tokens or using larger model
```

#### 6.3 Batch Size Validation
```gherkin
Scenario: Batch size exceeds worker capacity
  Given worker has 4 slots
  When user requests batch of 8 prompts
  Then validation fails
  And suggests splitting into 2 batches
```

---

## 7. Inference Execution (130-inference-execution.feature)

### ✅ What's Covered:
- Basic inference flow
- SSE streaming
- Token generation

### ❌ Critical Gaps:

#### 7.1 Concurrent Inference Requests
```gherkin
Scenario: Multiple requests to same worker
  Given worker has 4 slots
  When 4 requests arrive simultaneously
  Then all 4 are processed concurrently
  And slot allocation is tracked
  And 5th request waits for slot to free
```

#### 7.2 Request Cancellation
```gherkin
Scenario: User cancels inference mid-generation
  Given inference is generating tokens
  When user presses Ctrl+C
  Then rbee-keeper sends cancellation to worker
  And worker stops generation immediately
  And partial output is returned
  And slot is freed
```

#### 7.3 Generation Timeout
```gherkin
Scenario: Inference exceeds max generation time
  Given max_time is 30 seconds
  When generation takes 35 seconds
  Then worker stops generation
  And returns partial output
  And error "GENERATION_TIMEOUT" is logged
```

#### 7.4 EOS Token Handling
```gherkin
Scenario: Model generates EOS token early
  Given max_tokens is 100
  When model generates EOS at token 45
  Then generation stops at token 45
  And only 45 tokens are returned
  And finish_reason is "stop" (not "length")
```

---

## 8. End-to-End Flows (160-end-to-end-flows.feature)

### ✅ What's Covered:
- Full inference flow
- Multi-node scenarios

### ❌ Critical Gaps:

#### 8.1 Load Balancing
```gherkin
Scenario: Distribute requests across multiple workers
  Given 3 workers are available
  When 10 requests arrive
  Then requests are distributed evenly
  And no worker is overloaded
  And total throughput is maximized
```

#### 8.2 Failover
```gherkin
Scenario: Worker crashes during request processing
  Given worker-001 is processing request
  When worker-001 crashes
  Then queen-rbee detects crash
  And request is retried on worker-002
  And user receives result without error
```

#### 8.3 Rolling Updates
```gherkin
Scenario: Update worker without downtime
  Given worker-001 is serving requests
  When new worker-002 starts with updated model
  Then new requests go to worker-002
  And worker-001 completes in-flight requests
  And worker-001 is gracefully shutdown
```

---

## 9. Missing Feature Files

### 9.1 Observability & Monitoring
```gherkin
Feature: Metrics and Monitoring
  Scenario: Prometheus metrics export
  Scenario: Request tracing with trace IDs
  Scenario: Performance profiling
  Scenario: Resource usage dashboards
```

### 9.2 Configuration Management
```gherkin
Feature: Configuration Validation
  Scenario: Invalid YAML configuration
  Scenario: Configuration hot-reload
  Scenario: Environment variable overrides
  Scenario: Configuration schema validation
```

### 9.3 Security & Auth
```gherkin
Feature: Security Hardening
  Scenario: TLS certificate validation
  Scenario: mTLS between components
  Scenario: API key rotation
  Scenario: Rate limiting per API key
```

### 9.4 Disaster Recovery
```gherkin
Feature: Backup and Recovery
  Scenario: Catalog backup and restore
  Scenario: Registry state persistence
  Scenario: Crash recovery with state reconstruction
```

### 9.5 Performance Testing
```gherkin
Feature: Load Testing
  Scenario: 100 concurrent requests
  Scenario: Sustained load for 1 hour
  Scenario: Memory leak detection
  Scenario: Throughput benchmarking
```

---

## Priority Recommendations

### 🔴 P0 - Critical (Must Add):
1. **Concurrent worker scenarios** (race conditions)
2. **Worker failover and retry logic**
3. **Request cancellation**
4. **Catalog corruption recovery**

### 🟡 P1 - High (Should Add):
5. **Multi-GPU selection logic**
6. **Partial download resume**
7. **Registry split-brain resolution**
8. **Generation timeout handling**

### 🟢 P2 - Medium (Nice to Have):
9. **Load balancing scenarios**
10. **Rolling update scenarios**
11. **Metrics and monitoring**
12. **Configuration validation**

### 🔵 P3 - Low (Future):
13. **Security hardening**
14. **Performance benchmarking**
15. **Disaster recovery**

---

## Next Steps for TEAM-080

1. **Review this gap analysis** with product team
2. **Prioritize which gaps to address** based on risk
3. **Create new feature files** for P0 gaps
4. **Add missing scenarios** to existing files
5. **Wire up step definitions** for new scenarios

**Estimated Work:**
- P0 gaps: ~20 new scenarios = 40+ step definitions
- P1 gaps: ~15 new scenarios = 30+ step definitions
- Total: ~35 scenarios = 70+ step definitions

---

## Conclusion

The existing feature files provide a **solid foundation** for happy-path testing, but **lack critical edge cases** that will occur in production:

- **Concurrency** - No race condition testing
- **Failure recovery** - Limited failover scenarios
- **Performance** - No load/stress testing
- **Security** - Minimal auth/security testing

**Recommendation:** Address P0 gaps before v1.0 release to avoid production incidents.

---

**Analysis by:** TEAM-079  
**Date:** 2025-10-11  
**Status:** Ready for review 🔍
