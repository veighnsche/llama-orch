# TEAM-218: rbee-hive Behavior Investigation

**Phase:** 1 (Main Binaries)  
**Component:** `20_rbee_hive` - Hive daemon (manages workers)  
**Duration:** 1 day  
**Output:** `TEAM_218_RBEE_HIVE_BEHAVIORS.md`

---

## Mission

Inventory ALL behaviors in `rbee-hive` daemon to enable comprehensive test coverage.

---

## Investigation Areas

### 1. HTTP API Surface

**File:** `bin/20_rbee_hive/src/main.rs` + router files

**Tasks:**
- Document ALL HTTP endpoints
- Document request/response schemas
- Document status codes
- Document error responses

**Endpoints to Document:**
- `/health` - Health check
- `/heartbeat` - Heartbeat to queen
- `/capabilities` - Device/GPU capabilities
- `/v1/workers/*` - Worker management
- `/v1/models/*` - Model management
- `/v1/downloads/*` - Download tracking
- Any other endpoints

### 2. Worker Lifecycle Management

**Files:**
- Look for worker-lifecycle crate usage
- Look for worker-registry crate usage
- Worker operation handlers

**Tasks:**
- Document worker spawn behavior
- Document worker stop behavior
- Document worker status tracking
- Document worker heartbeat handling
- Document worker crash detection

**Operations to Document:**
- WorkerList
- WorkerStart
- WorkerStop
- WorkerStatus
- WorkerGet

**Edge Cases:**
- Worker fails to start
- Worker crashes mid-inference
- Worker hangs/freezes
- Multiple workers on same GPU
- VRAM exhaustion

### 3. Model Provisioning

**Files:**
- Look for model-provisioner crate usage
- Look for model-catalog crate usage
- Look for download-tracker crate usage

**Tasks:**
- Document model download flow
- Document model validation
- Document model caching
- Document download progress tracking
- Document download failure handling

**Behaviors:**
- How models are discovered
- How downloads are initiated
- How progress is tracked
- How completed downloads are verified
- How failures are retried

### 4. Device Detection

**Files:**
- Look for device-detection crate usage
- GPU detection logic
- VRAM checking logic

**Tasks:**
- Document GPU detection flow
- Document VRAM availability checks
- Document device capability reporting
- Document CUDA availability detection
- Document fallback to CPU

**Critical Questions:**
- What happens if no GPU?
- What happens if CUDA unavailable?
- How is VRAM tracked?
- How are device changes detected?

### 5. Heartbeat System

**Files:**
- Look for heartbeat to queen-rbee
- Look for heartbeat crate usage

**Tasks:**
- Document heartbeat frequency
- Document heartbeat payload
- Document heartbeat failure handling
- Document registration flow

**Behaviors:**
- When are heartbeats sent?
- What data is included?
- What happens if queen unreachable?
- How is heartbeat state managed?

### 6. Capabilities Endpoint

**Files:**
- `/capabilities` endpoint handler
- Capabilities caching logic

**Tasks:**
- Document capabilities discovery
- Document capabilities caching
- Document capabilities refresh
- Document timeout handling

**Critical:**
- How often are capabilities refreshed?
- What if device detection times out?
- What if GPU becomes unavailable?

### 7. Configuration Management

**Files:**
- Look for RbeeConfig usage
- Look for environment variables
- Look for worker configs

**Tasks:**
- Document ALL configuration sources
- Document config validation
- Document default values
- Document worker-specific config

**Configuration:**
- Queen URL
- Worker binary paths
- Model cache directory
- GPU allocation policy
- Timeout values

### 8. Daemon Lifecycle

**Files:**
- `bin/20_rbee_hive/src/main.rs` (startup/shutdown)

**Tasks:**
- Document startup sequence
- Document initialization steps
- Document graceful shutdown
- Document worker cleanup on shutdown
- Document signal handling

**Behaviors:**
- Port binding
- Config loading
- HTTP server startup
- Device detection at startup
- Queen registration
- Shutdown worker cleanup

### 9. Download Tracking

**Files:**
- Look for download-tracker crate usage

**Tasks:**
- Document active download tracking
- Document download progress reporting
- Document download completion detection
- Document download cleanup

**Questions:**
- How are downloads tracked?
- How is progress reported?
- What happens on download failure?
- How are partial downloads handled?

### 10. Monitor Integration

**Files:**
- Look for monitor crate usage
- System monitoring logic

**Tasks:**
- Document what is monitored
- Document monitoring frequency
- Document alert thresholds
- Document failure detection

---

## Investigation Methodology

### Step 1: Read Main Entry Point
```bash
cat bin/20_rbee_hive/src/main.rs
```

### Step 2: Identify All Modules
```bash
find bin/20_rbee_hive/src -name "*.rs"
```

### Step 3: Check Dependencies
```bash
cat bin/20_rbee_hive/Cargo.toml
```

### Step 4: Examine Supporting Crates
```bash
# These crates are critical to hive behavior
ls bin/25_rbee_hive_crates/
```

### Step 5: Check Existing Tests
```bash
find bin/20_rbee_hive -name "*test*.rs"
find bin/20_rbee_hive/bdd -name "*.feature"
```

---

## Key Files to Investigate

1. `bin/20_rbee_hive/src/main.rs` - Entry point, server setup
2. `bin/20_rbee_hive/Cargo.toml` - Dependencies
3. HTTP route handlers
4. Worker management modules
5. Model provisioning modules
6. Heartbeat modules
7. Capabilities modules

---

## Expected Behaviors to Document

### HTTP API Behaviors
- [ ] All endpoints documented
- [ ] Request/response schemas
- [ ] Error responses
- [ ] Status codes

### Worker Management Behaviors
- [ ] Worker spawn logic
- [ ] Worker stop logic
- [ ] Worker status tracking
- [ ] Worker heartbeats
- [ ] Worker crash handling

### Model Management Behaviors
- [ ] Model discovery
- [ ] Model download
- [ ] Download progress tracking
- [ ] Download failure handling
- [ ] Model validation

### Device Detection Behaviors
- [ ] GPU detection
- [ ] VRAM checking
- [ ] CUDA detection
- [ ] Capability reporting
- [ ] Fallback to CPU

### Heartbeat Behaviors
- [ ] Heartbeat to queen
- [ ] Heartbeat frequency
- [ ] Heartbeat payload
- [ ] Failure handling
- [ ] Registration flow

### Configuration Behaviors
- [ ] Config loading
- [ ] Config validation
- [ ] Default values
- [ ] Worker configs

### Daemon Behaviors
- [ ] Startup sequence
- [ ] Graceful shutdown
- [ ] Worker cleanup
- [ ] Signal handling

---

## Deliverables Checklist

- [ ] All HTTP endpoints documented
- [ ] All worker operations documented
- [ ] All model operations documented
- [ ] All device detection documented
- [ ] All heartbeat behaviors documented
- [ ] All error paths documented
- [ ] All configuration documented
- [ ] Daemon lifecycle documented
- [ ] Existing test coverage assessed
- [ ] Coverage gaps identified
- [ ] Code signatures added (`// TEAM-218: Investigated`)
- [ ] Document follows template
- [ ] Document ≤3 pages
- [ ] Examples include line numbers

---

## Success Criteria

1. ✅ Complete behavior inventory document
2. ✅ All HTTP APIs documented
3. ✅ All worker lifecycle documented
4. ✅ All model provisioning documented
5. ✅ All device detection documented
6. ✅ All heartbeat flows documented
7. ✅ Test coverage gaps identified
8. ✅ Code signatures added
9. ✅ No TODO markers in document

---

## Critical Focus Areas

### 1. Worker Lifecycle
Complex state machine - document thoroughly:
- Spawn → Running → Stopped
- Crash detection
- Restart logic
- Cleanup on shutdown

### 2. Model Provisioning
Multi-step process - document all steps:
- Discovery → Download → Validation → Cache
- Progress tracking
- Failure handling
- Retry logic

### 3. Device Detection
Hardware-dependent - document edge cases:
- No GPU available
- CUDA not installed
- VRAM exhausted
- Device changes

### 4. Heartbeat to Queen
Critical for orchestration:
- Registration flow
- Heartbeat frequency
- Failure handling
- Reconnection logic

---

## Next Steps After Completion

1. Hand off to TEAM-242 for test plan creation
2. Document will be used to create:
   - Unit test plan
   - BDD test plan
   - Integration test plan
   - E2E test plan

---

**Status:** READY  
**Blocked By:** None (can start immediately)  
**Blocks:** TEAM-242 (test planning)
