# Contract Dependency Analysis

**Date:** Oct 24, 2025  
**Author:** TEAM-285

## Contract Hierarchy

```
shared-contract (foundation)
    ↓
    ├─→ worker-contract
    └─→ hive-contract

operations-contract (independent)
```

## 1. shared-contract

**Purpose:** Common types for ALL components (workers + hives)

**Exports:**
- `HealthStatus` (Healthy/Degraded/Unhealthy)
- `OperationalStatus` (Starting/Ready/Busy/Stopping/Stopped)
- `HeartbeatTimestamp` with `is_recent()` helper
- `HeartbeatPayload` trait
- Constants (30s interval, 90s timeout)

**Consumers:**
- ✅ worker-contract (imports shared-contract)
- ✅ hive-contract (imports shared-contract)

**Does NOT import:** operations-contract ✅ (correct - different concern)

---

## 2. worker-contract

**Purpose:** Worker-specific types and heartbeat contract

**Imports:**
- ✅ shared-contract (for HealthStatus, OperationalStatus, HeartbeatPayload)

**Exports:**
- `WorkerInfo` - Full worker details
- `WorkerStatus` - Worker-specific status enum
- `WorkerHeartbeat` - Implements HeartbeatPayload trait

**Consumers:**
- ✅ llm-worker-rbee (sends heartbeats with WorkerInfo)
- ✅ queen-rbee (worker-registry uses WorkerHeartbeat)

**Does NOT import:** operations-contract ✅ (correct - heartbeats ≠ operations)

---

## 3. hive-contract

**Purpose:** Hive-specific types and heartbeat contract

**Imports:**
- ✅ shared-contract (for HealthStatus, OperationalStatus, HeartbeatPayload)

**Exports:**
- `HiveInfo` - Full hive details
- `HiveHeartbeat` - Implements HeartbeatPayload trait

**Consumers:**
- ✅ rbee-hive (sends heartbeats with HiveInfo)
- ✅ queen-rbee (hive-registry uses HiveHeartbeat)

**Does NOT import:** operations-contract ✅ (correct - heartbeats ≠ operations)

---

## 4. operations-contract

**Purpose:** Job operations contract (queen ↔ hive communication)

**Imports:**
- ❌ shared-contract (NO - and this is CORRECT!)
- ❌ worker-contract (NO - and this is CORRECT!)
- ❌ hive-contract (NO - and this is CORRECT!)

**Why no contract imports?**
Operations are about **job routing**, not **component health/status**. These are orthogonal concerns:
- **Heartbeats** (contracts) = "I'm alive, here's my status"
- **Operations** = "Do this work: spawn worker, download model, run inference"

**Exports:**
- `Operation` enum (all job types)
- Request types (WorkerSpawnRequest, ModelDownloadRequest, InferRequest, etc.)
- Response types (WorkerSpawnResponse, ModelListResponse, etc.)
- API spec (HiveApiSpec with endpoint constants)

**Consumers:**
- ✅ rbee-keeper (creates operations from CLI)
- ✅ queen-rbee (routes operations)
- ✅ rbee-hive (executes operations)
- ✅ job-client (submits operations via HTTP)

---

## Job Infrastructure Crates

### job-server (Generic Job Registry)

**Purpose:** Generic job tracking (works with ANY payload type)

**Imports:**
- ❌ operations-contract (NO - deliberately generic!)
- ❌ worker-contract (NO)
- ❌ hive-contract (NO)

**Why no imports?**
job-server is **generic over payload type** - it doesn't need to know what kind of jobs it's tracking. This is good design!

```rust
pub struct JobRegistry<T> {  // Generic!
    jobs: HashMap<String, Job<T>>,
}
```

**Consumers:**
- ✅ llm-worker-rbee (tracks inference jobs)
- ✅ queen-rbee (tracks orchestration jobs)
- ✅ rbee-hive (tracks worker lifecycle jobs)

---

### job-client (HTTP Job Submission)

**Purpose:** Submit operations and stream results via SSE

**Imports:**
- ✅ operations-contract (needs Operation enum to submit jobs)

**Why import operations-contract?**
job-client is specifically for submitting **operations** to queen/hive, so it needs to know about the Operation types.

**Consumers:**
- ✅ queen-rbee (hive_forwarder uses job-client to forward ops to hive)
- ✅ rbee-keeper (could use job-client but currently has own implementation)

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    shared-contract                          │
│              (Common types for all components)              │
│                                                             │
│  • HealthStatus, OperationalStatus                         │
│  • HeartbeatTimestamp, HeartbeatPayload trait              │
│  • Constants, ContractError                                │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ↓                          ↓
    ┌──────────────────┐      ┌──────────────────┐
    │ worker-contract  │      │  hive-contract   │
    │                  │      │                  │
    │ • WorkerInfo     │      │ • HiveInfo       │
    │ • WorkerHeartbeat│      │ • HiveHeartbeat  │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             ↓                         ↓
    ┌──────────────────┐      ┌──────────────────┐
    │ llm-worker-rbee  │      │    rbee-hive     │
    │                  │      │                  │
    │ Sends heartbeats │      │ Sends heartbeats │
    └──────────────────┘      └──────────────────┘
             │                         │
             └────────────┬────────────┘
                          ↓
                  ┌───────────────┐
                  │  queen-rbee   │
                  │               │
                  │ Tracks both   │
                  │ via registries│
                  └───────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  operations-contract                        │
│         (Job operations - independent of heartbeats)        │
│                                                             │
│  • Operation enum                                          │
│  • Request/Response types                                  │
│  • API specification                                       │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ↓                          ↓
    ┌──────────────────┐      ┌──────────────────┐
    │   rbee-keeper    │      │    job-client    │
    │                  │      │                  │
    │ Creates ops      │      │ Submits ops      │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             └────────────┬────────────┘
                          ↓
                  ┌───────────────┐
                  │  queen-rbee   │
                  │               │
                  │ Routes ops    │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │   rbee-hive   │
                  │               │
                  │ Executes ops  │
                  └───────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     job-server                              │
│              (Generic - no contract imports!)               │
│                                                             │
│  • JobRegistry<T> - Generic over payload type              │
│  • Used by: worker, queen, hive                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Are Crates Synergizing?

### ✅ YES - Excellent Separation of Concerns

**Heartbeat System (Contracts):**
- shared-contract → worker-contract → llm-worker-rbee → queen-rbee
- shared-contract → hive-contract → rbee-hive → queen-rbee
- **Purpose:** Component health tracking
- **Flow:** Workers/Hives send heartbeats TO queen

**Operations System (Operations Contract):**
- operations-contract → rbee-keeper → queen-rbee → rbee-hive
- operations-contract → job-client (used by queen)
- **Purpose:** Job execution
- **Flow:** Keeper creates ops → Queen routes → Hive executes

**Job Tracking (Generic Infrastructure):**
- job-server (generic, no contract imports)
- **Purpose:** Track job state
- **Used by:** All three binaries (worker, queen, hive)

---

## Key Design Insights

### 1. ✅ Operations ≠ Heartbeats (Orthogonal Concerns)

**operations-contract does NOT import shared-contract** because:
- Operations are about **work requests** (spawn, download, infer)
- Heartbeats are about **component status** (alive, healthy, ready)
- These are separate concerns and should stay separate!

### 2. ✅ job-server is Generic (No Contract Imports)

**job-server does NOT import any contracts** because:
- It's generic over payload type `JobRegistry<T>`
- Works with ANY job type (inference, lifecycle, etc.)
- This makes it reusable across all binaries

### 3. ✅ job-client Knows About Operations

**job-client DOES import operations-contract** because:
- Its specific purpose is to submit Operation types
- It's not generic - it's specifically for the queen↔hive protocol
- This is correct specialization

### 4. ✅ Workers Use worker-contract for Heartbeats

**llm-worker-rbee imports worker-contract** to:
- Build WorkerInfo with full details
- Send WorkerHeartbeat to queen
- Align with queen's worker-registry expectations

---

## Conclusion

✅ **All crates are synergizing correctly!**

**Separation of Concerns:**
- Heartbeats (contracts) ≠ Operations (operations-contract)
- Generic infrastructure (job-server) ≠ Specific protocols (job-client)

**Type Safety:**
- Compile-time guarantees at every boundary
- No JSON guessing
- Clear contracts between components

**Maintainability:**
- Each crate has a single, clear purpose
- Dependencies flow in one direction (no cycles)
- Easy to understand and modify

**No Missing Links:**
- llm-worker-rbee ✅ uses worker-contract
- rbee-hive ✅ uses hive-contract
- queen-rbee ✅ uses both registries
- All operations ✅ use operations-contract
- Job tracking ✅ uses generic job-server

**The architecture is sound!** 🎉
