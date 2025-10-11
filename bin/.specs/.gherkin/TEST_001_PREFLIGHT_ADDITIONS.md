# Test-001.md Preflight Additions
# Updated by: TEAM-077
# Date: 2025-10-11
# Status: ✅ ALL MISSING FEATURES ADDED

## What Was Missing

The test-001.md document was missing **3 critical M1 components**:
1. SSH preflight checks (Phase 2a)
2. rbee-hive preflight checks (Phase 3a)
3. Worker binaries catalog (Phase 3a)

## What Was Added

### 1. Phase 2a: SSH Preflight Checks (NEW - M1) ✅

**Location:** Between Phase 2 and Phase 2b
**Feature File:** 050-ssh-preflight-checks.feature

**Checks Performed:**
1. SSH connection reachable - Test basic connectivity
2. SSH authentication works - Verify key-based auth
3. SSH command execution - Test `echo 'test'` command
4. Network latency acceptable - Check latency < 100ms
5. rbee-hive binary exists - Verify binary at install_path

**Error Scenarios:**
- EH-001a: SSH connection timeout (already documented)
- EH-001b: SSH authentication failure (already documented)
- EH-001c: SSH command execution failure (already documented)

**Narration Flow:**
```
[queen-rbee] 🔍 Running SSH preflight checks for workstation
[queen-rbee] ✅ SSH connectivity: OK
[queen-rbee] ✅ SSH authentication: OK
[queen-rbee] ✅ Command execution: OK
[queen-rbee] ✅ Network latency: 15ms
[queen-rbee] ✅ rbee-hive binary: Found at /home/vince/rbee/target/release/rbee-hive
[queen-rbee] ✅ All SSH preflight checks passed
```

**Failure Example:**
```
[queen-rbee] ❌ SSH preflight failed: rbee-hive binary not found
Suggestion: Install rbee-hive: rbee-keeper setup install --node workstation
Exit code: 1
```

### 2. Phase 3a: rbee-hive Preflight Checks (NEW - M1) ✅

**Location:** Between Phase 3 and Phase 4
**Feature File:** 060-rbee-hive-preflight-checks.feature

**Checks Performed:**
1. HTTP API responding - GET /v1/health returns 200
2. Version compatibility - rbee-hive version compatible with queen-rbee
3. Worker binaries available - Check worker_binaries catalog
4. Backend catalog populated - CUDA/Metal/CPU detected
5. Sufficient resources - RAM, disk space available

**New API Endpoints:**
```
GET http://workstation.home.arpa:9200/v1/worker-binaries/list
GET http://workstation.home.arpa:9200/v1/backends/list
```

**Error Scenarios:**

**EH-019a: Worker Binary Not Installed**
- Trigger: Requested worker type not in binaries catalog
- Detection: Worker binary not found in catalog
- Response: FAIL FAST with installation suggestion
- Exit Code: 1
- Message: "Worker binary not found: llm-worker-rbee"
- Suggestion: "Install worker: rbee-keeper setup install --node workstation"

**EH-019b: rbee-hive Version Incompatible**
- Trigger: rbee-hive version too old/new
- Detection: Version check fails
- Response: Error with upgrade suggestion
- Exit Code: 1
- Message: "rbee-hive version incompatible: need >=0.1.0, have 0.0.9"
- Suggestion: "Upgrade rbee-hive: rbee-keeper setup install --node workstation"

**Narration Flow:**
```
[queen-rbee] 🔍 Running rbee-hive preflight checks
[queen-rbee] ✅ HTTP API: Responding
[queen-rbee] ✅ Version: 0.1.0 (compatible)
[queen-rbee] ✅ Worker binaries: llm-worker-rbee found
[queen-rbee] ✅ Backends: cpu, cuda available
[queen-rbee] ✅ Resources: Sufficient
[queen-rbee] ✅ All rbee-hive preflight checks passed
```

**Failure Example:**
```
[queen-rbee] ❌ rbee-hive preflight failed: llm-worker-rbee not installed
Suggestion: Install worker: rbee-keeper setup install --node workstation
Exit code: 1
```

### 3. Worker Binaries Catalog (NEW - M1) ✅

**Location:** Part of Phase 3a
**Feature File:** 025-worker-binaries-catalog.feature

**Purpose:** Track which worker types are installed on each rbee-hive

**API Endpoint:**
```
GET http://workstation.home.arpa:9200/v1/worker-binaries/list
```

**Response Format:**
```json
{
  "binaries": [
    {
      "worker_type": "llm-worker-rbee",
      "version": "0.1.0",
      "binary_path": "/home/vince/rbee/target/release/llm-worker-rbee",
      "installed_at_unix": 1728508000
    }
  ]
}
```

**Database Schema (SQLite):**
```sql
CREATE TABLE worker_binaries (
    id INTEGER PRIMARY KEY,
    worker_type TEXT NOT NULL,  -- llm-worker-rbee, sd-worker-rbee, etc.
    version TEXT NOT NULL,
    binary_path TEXT NOT NULL,
    installed_at_unix INTEGER,
    UNIQUE(worker_type)
);
```

**Operations:**
- Register worker binary after installation
- Query available worker types
- Version compatibility check
- Verify binary exists before spawning

## Updated Phase Flow

**Complete flow with all preflight checks:**

1. **Phase 0:** queen-rbee loads rbee-hive registry
2. **Phase 1:** rbee-keeper → queen-rbee (task submission)
3. **Phase 2:** queen-rbee → rbee-hive (SSH)
   - **Phase 2a:** SSH Preflight Checks ✅ NEW!
   - **Phase 2b:** Start rbee-hive via SSH
4. **Phase 3:** queen-rbee checks worker registry
   - **Phase 3a:** rbee-hive Preflight Checks ✅ NEW!
     - Worker Binaries Catalog Check ✅ NEW!
     - Backend Catalog Check
5. **Phase 4:** queen-rbee → rbee-hive: Spawn worker
6. **Phase 5:** rbee-hive checks model catalog
7. **Phase 6:** rbee-hive downloads model
8. **Phase 7:** rbee-hive registers model in catalog
9. **Phase 8:** rbee-hive worker preflight (RAM, VRAM, disk, backend)
10. **Phase 9:** rbee-hive spawns worker
11. **Phase 10:** rbee-hive registers worker
12. **Phase 11:** rbee-hive returns worker URL to queen-rbee
13. **Phase 12:** queen-rbee returns worker URL to rbee-keeper
14. **Phase 13:** rbee-keeper → worker: Execute inference
15. **Phase 14:** Cascading shutdown

## Feature File Mapping Updated

**Updated mapping with phase references:**

- 010-ssh-registry-management.feature (Phase 0)
- 020-model-catalog.feature (Phase 5-7)
- 025-worker-binaries-catalog.feature (Phase 3a) ✅ NEW!
- 040-rbee-hive-worker-registry.feature (Phase 3, 10)
- 050-ssh-preflight-checks.feature (Phase 2a) ✅ NEW!
- 060-rbee-hive-preflight-checks.feature (Phase 3a) ✅ NEW!
- 070-worker-preflight-checks.feature (Phase 8)
- 080-worker-rbee-lifecycle.feature (Phase 9)
- 090-rbee-hive-lifecycle.feature (Phase 2b, 14)
- 110-inference-execution.feature (Phase 13)
- 120-input-validation.feature
- 130-cli-commands.feature
- 140-end-to-end-flows.feature (All phases)

## Why These Were Missing

The original test-001.md document focused on the **happy path flow** and didn't include the **preflight validation layers** that are critical for M1:

1. **SSH preflight** - Needed to validate SSH before attempting to start rbee-hive
2. **rbee-hive preflight** - Needed to validate rbee-hive readiness before spawning workers
3. **Worker binaries catalog** - Needed to track which worker types are installed

These are **M1 requirements** for robust pool manager lifecycle management.

## Document Completeness

**test-001.md now includes:**
- ✅ All 3 preflight check levels (SSH, rbee-hive, worker)
- ✅ Worker binaries catalog
- ✅ Backend catalog checks
- ✅ New error scenarios (EH-019a, EH-019b)
- ✅ Complete narration flows
- ✅ Phase-by-phase mapping to feature files
- ✅ All M1 components documented

**Status:** ✅ COMPLETE - All missing M1 features added to test-001.md

---

**TEAM-077 says:** All missing preflight checks added! Worker binaries catalog documented! Phase flow complete! Feature file mapping updated! test-001.md is now comprehensive for M1! 🐝
