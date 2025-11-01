# TEAM-381: Types That Should Come From Rust

**Date:** 2025-11-01  
**Status:** 📋 AUDIT

## Overview

This document identifies all TypeScript types that are currently manually defined but should be auto-generated from Rust using `tsify`.

## Current Status

### ✅ Already Migrated (Hive)
- `ModelInfo` - Auto-generated from `operations-contract`

### 🔴 Needs Migration

## 1. Heartbeat & Telemetry Types

### Location
- **TypeScript:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts`
- **TypeScript:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/index.ts`
- **Rust:** `bin/25_rbee_hive_crates/monitor/src/lib.rs`
- **Rust:** `bin/97_contracts/hive-contract/src/heartbeat.rs`

### Types to Migrate

#### ProcessStats
**Current TypeScript (manual):**
```typescript
// bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts
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

**Rust Source:**
```rust
// bin/25_rbee_hive_crates/monitor/src/lib.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
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

**Action:** Add `#[cfg_attr(feature = "wasm", derive(Tsify))]` to Rust struct

#### HiveDevice
**Current TypeScript (missing!):**
```typescript
// Should be added
export interface HiveDevice {
  id: string
  name: string
  device_type: string
  vram_gb?: number
  compute_capability?: string
}
```

**Rust Source:**
```rust
// bin/97_contracts/hive-contract/src/heartbeat.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveDevice {
    pub id: String,
    pub name: String,
    pub device_type: String,
    pub vram_gb: Option<u32>,
    pub compute_capability: Option<String>,
}
```

**Action:** Add `#[cfg_attr(feature = "wasm", derive(Tsify))]` and export

#### HiveTelemetry
**Current TypeScript (manual):**
```typescript
// bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts
export interface HiveTelemetry {
  type: 'hive_telemetry'
  hive_id: string
  timestamp: string
  workers: ProcessStats[]
}
```

**Rust Source:**
```rust
// Should be in hive-contract or telemetry-registry
// Currently seems to be constructed manually
```

**Action:** Create Rust struct and add Tsify

#### QueenHeartbeat
**Current TypeScript (manual):**
```typescript
// bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts
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
```

**Rust Source:**
```rust
// Should be in queen-rbee contracts
// Currently seems to be constructed manually
```

**Action:** Create Rust struct and add Tsify

## 2. Hive Info Types

### HiveInfo
**Current TypeScript (manual):**
```typescript
// bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/index.ts
export interface HiveInfo {
  id: string
  hostname: string
  port: number
  operational_status: string
  health_status: {
    status: string
  }
  version: string
}
```

**Rust Source:**
```rust
// bin/97_contracts/hive-contract/src/types.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveInfo {
    pub id: String,
    pub hostname: String,
    pub port: u16,
    pub operational_status: OperationalStatus,
    pub health_status: HealthStatus,
    pub version: String,
}
```

**Action:** Add `#[cfg_attr(feature = "wasm", derive(Tsify))]` to Rust struct

## 3. Orchestrator API Types

### Location
- **Rust:** `contracts/api-types/src/generated.rs`
- **TypeScript:** Currently missing in frontend!

### Types to Add

#### TaskRequest
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task_id: String,
    pub session_id: String,
    pub workload: Workload,
    pub engine: Engine,
    // ... more fields
}
```

#### AdmissionResponse
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionResponse {
    pub task_id: String,
    pub queue_position: i32,
    pub predicted_start_ms: i64,
    pub streams: AdmissionStreams,
    pub preparation: Preparation,
}
```

**Action:** Add tsify support to `contracts/api-types`

## 4. Operations Contract Types

### Location
- **Rust:** `bin/97_contracts/operations-contract/src/`

### Already Has Tsify Support
- ✅ `ModelInfo` (done)

### Needs Tsify

#### WorkerProcessInfo
```rust
// bin/97_contracts/operations-contract/src/responses.rs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerProcessInfo {
    pub pid: u32,
    pub worker_id: String,
    pub model: String,
    pub port: u16,
    pub status: String,
}
```

#### ModelListResponse
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelListResponse {
    pub models: Vec<ModelInfo>,
}
```

#### WorkerSpawnResponse
```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerSpawnResponse {
    pub worker_id: String,
    pub port: u16,
    pub pid: u32,
    pub status: String,
}
```

**Action:** Add `#[cfg_attr(feature = "wasm", derive(Tsify))]` to all response types

## Migration Priority

### Phase 1: Critical (Do Now)
1. **ProcessStats** - Used in both Queen and Hive UIs
2. **HiveInfo** - Core type for hive management
3. **ModelInfo** - ✅ Already done

### Phase 2: Important (Next)
4. **WorkerProcessInfo** - Worker management
5. **HiveDevice** - Device capabilities
6. **Response types** - All operations-contract responses

### Phase 3: Nice to Have
7. **QueenHeartbeat** - Queen-specific telemetry
8. **HiveTelemetry** - Hive-specific telemetry
9. **Orchestrator types** - TaskRequest, AdmissionResponse, etc.

## Implementation Plan

### Step 1: Add tsify to monitor crate
```toml
# bin/25_rbee_hive_crates/monitor/Cargo.toml
[dependencies]
tsify = { version = "0.4", optional = true }
wasm-bindgen = { version = "0.2", optional = true }

[features]
wasm = ["tsify", "wasm-bindgen"]
```

### Step 2: Annotate ProcessStats
```rust
// bin/25_rbee_hive_crates/monitor/src/lib.rs
#[cfg(feature = "wasm")]
use tsify::Tsify;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ProcessStats {
    // ... fields
}
```

### Step 3: Add tsify to hive-contract
```toml
# bin/97_contracts/hive-contract/Cargo.toml
[dependencies]
tsify = { version = "0.4", optional = true }
wasm-bindgen = { version = "0.2", optional = true }

[features]
wasm = ["tsify", "wasm-bindgen"]
```

### Step 4: Annotate hive-contract types
```rust
// bin/97_contracts/hive-contract/src/heartbeat.rs
#[cfg(feature = "wasm")]
use tsify::Tsify;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct HiveDevice {
    // ... fields
}
```

### Step 5: Enable wasm features in SDKs
```toml
# bin/10_queen_rbee/ui/packages/queen-rbee-sdk/Cargo.toml
[dependencies]
rbee-hive-monitor = { path = "../../../../25_rbee_hive_crates/monitor", features = ["wasm"] }
hive-contract = { path = "../../../../97_contracts/hive-contract", features = ["wasm"] }
```

### Step 6: Re-export in SDK lib.rs
```rust
// bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/lib.rs
pub use rbee_hive_monitor::ProcessStats;
pub use hive_contract::{HiveDevice, HiveInfo};
```

### Step 7: Re-export in TypeScript
```typescript
// bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/index.ts
export type { 
  ProcessStats,
  HiveDevice,
  HiveInfo,
} from './pkg/bundler/rbee_sdk'
```

### Step 8: Remove manual TypeScript types
```typescript
// Delete manual definitions
// export interface ProcessStats { ... } ❌
// export interface HiveInfo { ... } ❌
```

## Benefits

### Single Source of Truth
- Types defined once in Rust
- TypeScript types generated automatically
- No drift between backend and frontend

### Type Safety
- Rust compiler enforces correctness
- TypeScript gets exact same types
- Compile-time errors catch mismatches

### Maintainability
- Update types in one place
- TypeScript updates automatically
- No manual synchronization

## Estimated Effort

| Phase | Types | Crates | Effort |
|-------|-------|--------|--------|
| Phase 1 | 3 types | 2 crates | 2 hours |
| Phase 2 | 6 types | 2 crates | 3 hours |
| Phase 3 | 9 types | 3 crates | 4 hours |
| **Total** | **18 types** | **3 crates** | **9 hours** |

## Success Criteria

- [ ] All manual TypeScript types removed
- [ ] All types auto-generated from Rust
- [ ] SDKs build successfully
- [ ] TypeScript types match Rust exactly
- [ ] No type drift possible
- [ ] Documentation updated

## Summary

**Current:** 18+ types manually duplicated in TypeScript  
**Goal:** All types auto-generated from Rust using tsify  
**Priority:** Start with ProcessStats, HiveInfo, and response types  
**Benefit:** Single source of truth, no drift, type safety  

**Next action:** Implement Phase 1 (ProcessStats, HiveInfo)
