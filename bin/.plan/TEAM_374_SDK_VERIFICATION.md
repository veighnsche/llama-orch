# TEAM-374: SDK Data Verification

**Date:** Oct 31, 2025  
**Status:** ✅ VERIFIED

---

## Question

Does the Queen SDK (WASM) receive the same data structure after Phase 3 changes?

**Answer:** ✅ YES - SDK types match backend exactly, no changes needed.

---

## Backend → SDK Data Flow

```
Hive → SSE → Queen → Aggregated SSE → SDK (WASM) → React UI
                ↓
        HeartbeatEvent enum
                ↓
        {type: "hive_telemetry", ...}
        {type: "queen", ...}
```

---

## Type Comparison

### Backend (Rust)

**File:** `bin/10_queen_rbee/src/http/heartbeat.rs`

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HeartbeatEvent {
    /// Hive telemetry with worker details
    HiveTelemetry {
        hive_id: String,
        timestamp: String,
        workers: Vec<ProcessStats>,
    },
    
    /// Queen's own heartbeat (sent every 2.5 seconds)
    Queen {
        workers_online: usize,
        workers_available: usize,
        hives_online: usize,
        hives_available: usize,
        worker_ids: Vec<String>,
        hive_ids: Vec<String>,
        timestamp: String,
    },
}
```

### SDK (TypeScript)

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts`

```typescript
// Hive telemetry event (sent every 1s from each hive)
export interface HiveTelemetry {
  type: 'hive_telemetry'
  hive_id: string
  timestamp: string
  workers: ProcessStats[]
}

// Queen heartbeat event (sent every 2.5s)
export interface QueenHeartbeat {
  type: 'queen'
  workers_online: number
  workers_available: number
  hives_online: number
  hives_available: number
  worker_ids: string[]
  hive_ids: string[]
  timestamp: string
}

// Union type for all heartbeat events
export type HeartbeatEvent = HiveTelemetry | QueenHeartbeat
```

### ✅ Verification: Types Match Perfectly

| Field | Backend (Rust) | SDK (TypeScript) | Match |
|-------|----------------|------------------|-------|
| `type` | `"hive_telemetry"` / `"queen"` | `"hive_telemetry"` / `"queen"` | ✅ |
| `hive_id` | `String` | `string` | ✅ |
| `timestamp` | `String` | `string` | ✅ |
| `workers` | `Vec<ProcessStats>` | `ProcessStats[]` | ✅ |
| `workers_online` | `usize` | `number` | ✅ |
| `workers_available` | `usize` | `number` | ✅ |
| `hives_online` | `usize` | `number` | ✅ |
| `hives_available` | `usize` | `number` | ✅ |
| `worker_ids` | `Vec<String>` | `string[]` | ✅ |
| `hive_ids` | `Vec<String>` | `string[]` | ✅ |

---

## ProcessStats Comparison

### Backend (Rust)

**File:** `bin/25_rbee_hive_crates/monitor/src/lib.rs`

```rust
pub struct ProcessStats {
    pub pid: u32,
    pub group: String,
    pub instance: String,
    pub cpu_pct: f64,
    pub rss_mb: u64,
    pub io_r_mb_s: f64,
    pub io_w_mb_s: f64,
    pub uptime_s: u64,
    pub gpu_util_pct: f64,
    pub vram_mb: u64,
    pub total_vram_mb: u64,
    pub model: Option<String>,
}
```

### SDK (TypeScript)

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts`

```typescript
export interface ProcessStats {
  pid: number
  group: string
  instance: string
  cpu_pct: number
  rss_mb: number
  io_r_mb_s: number
  io_w_mb_s: number
  uptime_s: number
  gpu_util_pct: number
  vram_mb: number
  total_vram_mb: number
  model: string | null
}
```

### ✅ Verification: ProcessStats Match Perfectly

| Field | Backend (Rust) | SDK (TypeScript) | Match |
|-------|----------------|------------------|-------|
| `pid` | `u32` | `number` | ✅ |
| `group` | `String` | `string` | ✅ |
| `instance` | `String` | `string` | ✅ |
| `cpu_pct` | `f64` | `number` | ✅ |
| `rss_mb` | `u64` | `number` | ✅ |
| `io_r_mb_s` | `f64` | `number` | ✅ |
| `io_w_mb_s` | `f64` | `number` | ✅ |
| `uptime_s` | `u64` | `number` | ✅ |
| `gpu_util_pct` | `f64` | `number` | ✅ |
| `vram_mb` | `u64` | `number` | ✅ |
| `total_vram_mb` | `u64` | `number` | ✅ |
| `model` | `Option<String>` | `string \| null` | ✅ |

---

## SDK Connection

### HeartbeatMonitor (WASM)

**File:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/heartbeat.rs`

```rust
// Connects to GET /v1/heartbeats/stream
let url = format!("{}/v1/heartbeats/stream", self.base_url);
let event_source = EventSource::new(&url)?;
```

**Endpoint:** `GET /v1/heartbeats/stream`  
**Status:** ✅ Still exists, unchanged

---

## What Changed in Phase 3?

### Backend Changes

1. ❌ **DELETED:** `POST /v1/hive-heartbeat` (old continuous POST)
2. ✅ **KEPT:** `GET /v1/heartbeats/stream` (SSE stream)
3. ✅ **KEPT:** `HeartbeatEvent` enum (unchanged)
4. ✅ **KEPT:** `ProcessStats` struct (unchanged)

### SDK Changes

**None!** The SDK was already using SSE (`/v1/heartbeats/stream`), not POST.

---

## Scheduler vs SDK Data

### Question: Does the scheduler see the same data as the SDK?

**Answer:** ✅ YES - Both use `TelemetryRegistry`

### Scheduler

**File:** `bin/15_queen_rbee_crates/scheduler/src/simple.rs`

```rust
pub fn new(worker_registry: Arc<TelemetryRegistry>) -> Self
```

**Data Source:** `TelemetryRegistry.list_online_workers()` → `Vec<ProcessStats>`

### SDK

**Data Source:** `GET /v1/heartbeats/stream` → `HeartbeatEvent::HiveTelemetry { workers: Vec<ProcessStats> }`

### Flow

```
Hive → SSE → Queen → TelemetryRegistry.update_workers()
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Scheduler           Broadcast SSE
         (queries registry)    (forwards to SDK)
                    ↓                   ↓
         find_best_worker()      HeartbeatEvent
              ↓                         ↓
        ProcessStats              ProcessStats
```

**Both see the same `ProcessStats` from `TelemetryRegistry`!**

---

## Test Verification

### Live Test Results

From TEAM_374_TEST_RESULTS.md:

```json
// Queen's SSE stream output:
event: heartbeat
data: {"type":"hive_telemetry","hive_id":"localhost","timestamp":"2025-10-31T12:44:41...","workers":[]}

event: heartbeat
data: {"type":"queen","workers_online":0,"workers_available":0,"hives_online":0,...}
```

**SDK receives exactly this format!**

---

## Conclusion

### ✅ SDK is Correctly Wired

1. ✅ SDK connects to `GET /v1/heartbeats/stream` (unchanged)
2. ✅ SDK types match backend types exactly
3. ✅ `HeartbeatEvent` enum unchanged
4. ✅ `ProcessStats` struct unchanged
5. ✅ Scheduler and SDK see the same data
6. ✅ Both use `TelemetryRegistry` as source of truth

### ✅ No SDK Changes Needed

The SDK was already using SSE, not POST. Phase 3 only deleted the POST endpoint that the SDK never used.

### ✅ Data Flow Verified

```
Hive → SSE → Queen.TelemetryRegistry
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   Scheduler              Broadcast SSE
   (queries)              (forwards)
        ↓                       ↓
  ProcessStats           HeartbeatEvent
  (for routing)          (for UI)
        ↓                       ↓
   Same data!            Same data!
```

---

**TEAM-374: SDK verification complete. Everything still works correctly!**
