# Revised Feature Structure - Consolidated & Worker Provisioning
# Created by: TEAM-077
# Date: 2025-10-11 14:35
# Status: REVISED BASED ON USER FEEDBACK

## User Feedback

1. **Preflight checks should be consolidated** - Not 3 separate features
2. **Need worker provisioning** - Build worker binaries from git (future: download)

## Problems with Previous Structure

### Too Fragmented (OLD):
- 025-worker-binaries-catalog.feature
- 050-ssh-preflight-checks.feature
- 060-rbee-hive-preflight-checks.feature
- 070-worker-preflight-checks.feature

**Issues:**
- 4 separate features for related validation concerns
- Binaries catalog mixed with preflight
- Doesn't cover worker binary building/provisioning

## Revised Structure (NEW)

### Consolidated Preflight Validation

**050-preflight-validation.feature** (Consolidates 3 features)
**Purpose:** ALL preflight checks in one logical feature
**Scenarios:**
- SSH connection validation (connection, auth, latency, binary exists)
- rbee-hive readiness validation (HTTP API, version, resources)
- Worker resource validation (RAM, VRAM, disk, backend)

**Why consolidated:**
- Preflight is ONE validation phase with 3 levels
- Logically grouped: "validate before proceeding"
- Easier to understand the complete validation flow

### Worker Provisioning (NEW!)

**025-worker-provisioning.feature** (NEW!)
**Purpose:** Build/download worker binaries and manage catalog
**Scenarios:**

**Building from Git (M1):**
- Check if worker binary exists in catalog
- If not found, build from git:
  ```bash
  cd /home/vince/rbee
  cargo build --release --bin llm-cuda-worker-rbee --features cuda
  ```
- Register built binary in catalog
- Verify binary works (test execution)

**Catalog Management:**
- List available worker types
- Query worker binary details
- Version compatibility checks
- Remove old worker binaries

**Future (M2+):**
- Download pre-built worker binaries
- Verify binary signatures
- Auto-update worker binaries

**Error Scenarios:**
- Worker binary not found, build fails
- Cargo build errors (missing dependencies)
- Binary verification fails
- Version incompatibility

## Complete M0-M1 Feature Files (12 files)

### Registries & Catalogs (3 files)
- **010-ssh-registry-management.feature** - Beehive registry (SSH details)
- **020-model-catalog.feature** - Model download and catalog
- **025-worker-provisioning.feature** - Build workers + binaries catalog ✅ REVISED!

### Validation & Preflight (1 file - consolidated!)
- **050-preflight-validation.feature** - ALL preflight checks ✅ CONSOLIDATED!
  - SSH preflight (connection, auth, latency, binary exists)
  - rbee-hive preflight (HTTP API, version, resources)
  - Worker preflight (RAM, VRAM, disk, backend)

### Registries (1 file)
- **060-rbee-hive-worker-registry.feature** - Local worker registry

### Daemon Lifecycles (3 files)
- **070-worker-rbee-lifecycle.feature** - worker-rbee daemon
- **080-rbee-hive-lifecycle.feature** - rbee-hive daemon
- **090-queen-rbee-lifecycle.feature** - queen-rbee daemon (M2 - defer)

### Execution & Operations (4 files)
- **100-inference-execution.feature** - Inference handling
- **110-input-validation.feature** - Input validation
- **120-cli-commands.feature** - CLI commands
- **130-end-to-end-flows.feature** - E2E integration

**Total M0-M1: 12 files** (was 13, consolidated 3 preflight into 1)

## Worker Provisioning Flow (NEW!)

### Phase 3b: Worker Provisioning (NEW!)

**Location:** After Phase 3a (rbee-hive preflight), before Phase 4 (spawn worker)

**Flow:**
1. queen-rbee checks: Does `llm-cuda-worker-rbee` exist in binaries catalog?
2. **If NO:** Provision worker binary
   - queen-rbee → rbee-hive: `POST /v1/worker-binaries/provision`
   - rbee-hive runs: `cargo build --release --bin llm-cuda-worker-rbee --features cuda`
   - rbee-hive registers binary in catalog
   - rbee-hive returns: binary_path, version
3. **If YES:** Use existing binary from catalog
4. Proceed to spawn worker (Phase 4)

**API Endpoint:**
```
POST http://workstation.home.arpa:9200/v1/worker-binaries/provision
{
  "worker_type": "llm-cuda-worker-rbee",
  "features": ["cuda"],
  "force_rebuild": false
}
```

**Response:**
```json
{
  "worker_type": "llm-cuda-worker-rbee",
  "version": "0.1.0",
  "binary_path": "/home/vince/rbee/target/release/llm-cuda-worker-rbee",
  "build_time_seconds": 120,
  "status": "ready"
}
```

**Narration:**
```
[rbee-hive] 🔍 Checking for llm-cuda-worker-rbee binary...
[rbee-hive] ❌ Binary not found in catalog
[rbee-hive] 🔨 Building llm-cuda-worker-rbee from git...
[rbee-hive] 📦 Running: cargo build --release --bin llm-cuda-worker-rbee --features cuda
[rbee-hive] ⏳ Building... (this may take 2-5 minutes)
[rbee-hive] ✅ Build complete! (120 seconds)
[rbee-hive] 📝 Registered in binaries catalog
[rbee-hive] ✅ Worker binary ready: /home/vince/rbee/target/release/llm-cuda-worker-rbee
```

**Error Scenarios:**

**EH-020a: Cargo Build Failed**
- **Trigger:** Cargo build returns non-zero exit code
- **Detection:** Build process fails
- **Response:** Error with build log
- **Exit Code:** 1
- **Message:** "Failed to build llm-cuda-worker-rbee"
- **Suggestion:** "Check build log: /home/vince/rbee/build.log"

**EH-020b: Missing Build Dependencies**
- **Trigger:** CUDA toolkit not installed for cuda feature
- **Detection:** Cargo reports missing dependency
- **Response:** Error with installation link
- **Exit Code:** 1
- **Message:** "Missing CUDA toolkit for cuda feature"
- **Suggestion:** "Install CUDA: https://developer.nvidia.com/cuda-downloads"

**EH-020c: Binary Verification Failed**
- **Trigger:** Built binary doesn't execute
- **Detection:** Test execution fails
- **Response:** Error with suggestion
- **Exit Code:** 1
- **Message:** "Built binary failed verification"
- **Suggestion:** "Try clean rebuild: cargo clean && cargo build"

### Future: Download Pre-Built Binaries (M2+)

**Instead of building from git:**
```
POST /v1/worker-binaries/provision
{
  "worker_type": "llm-cuda-worker-rbee",
  "source": "download",  // Instead of "build"
  "version": "0.1.0"
}
```

**rbee-hive downloads from artifact server:**
```bash
wget https://artifacts.rbee.dev/workers/llm-cuda-worker-rbee-0.1.0-linux-x86_64
chmod +x llm-cuda-worker-rbee
```

**Benefits:**
- Faster (no 2-5 minute build time)
- Consistent binaries (reproducible builds)
- Less disk space (no build artifacts)
- No build dependencies required

## Updated Phase Flow

**Complete flow with worker provisioning:**

1. **Phase 0:** queen-rbee loads rbee-hive registry
2. **Phase 1:** rbee-keeper → queen-rbee (task submission)
3. **Phase 2:** queen-rbee → rbee-hive (SSH)
4. **Phase 3:** Validation & Provisioning
   - **Phase 3a:** Preflight Validation (consolidated)
     - SSH preflight
     - rbee-hive preflight
     - Worker resource preflight
   - **Phase 3b:** Worker Provisioning ✅ NEW!
     - Check binaries catalog
     - Build from git if needed
     - Register in catalog
5. **Phase 4:** queen-rbee → rbee-hive: Spawn worker
6. **Phase 5:** rbee-hive checks model catalog
7. **Phase 6:** rbee-hive downloads model
8. **Phase 7:** rbee-hive registers model in catalog
9. **Phase 8:** rbee-hive spawns worker
10. **Phase 9:** rbee-hive registers worker
11. **Phase 10:** rbee-hive returns worker URL to queen-rbee
12. **Phase 11:** queen-rbee returns worker URL to rbee-keeper
13. **Phase 12:** rbee-keeper → worker: Execute inference
14. **Phase 13:** Cascading shutdown

## Benefits of Revised Structure

### 1. Logical Grouping ✅
- Preflight validation is ONE feature (not 3)
- Worker provisioning is separate concern
- Clearer separation of responsibilities

### 2. Complete Coverage ✅
- Now covers worker binary building
- Handles missing worker binaries
- Future-ready for binary downloads

### 3. Fewer Files ✅
- 12 files instead of 13
- Easier to navigate
- Less fragmentation

### 4. Better Names ✅
- "preflight-validation" is clearer than 3 separate files
- "worker-provisioning" clearly indicates building/downloading workers

## Summary

**Changes:**
- ✅ Consolidated 3 preflight features into 1
- ✅ Added worker provisioning (build from git)
- ✅ Added binaries catalog to worker provisioning
- ✅ Reduced from 13 to 12 M0-M1 files
- ✅ Added Phase 3b: Worker Provisioning
- ✅ Added error scenarios (EH-020a/b/c)
- ✅ Future-ready for binary downloads (M2+)

**Result:**
- More logical feature grouping
- Complete worker lifecycle (provision → spawn → execute)
- Clearer separation of concerns
- Ready for M1 implementation

---

**TEAM-077 says:** Preflight consolidated! Worker provisioning added! Build from git covered! Future binary downloads planned! Better structure achieved! 🐝
