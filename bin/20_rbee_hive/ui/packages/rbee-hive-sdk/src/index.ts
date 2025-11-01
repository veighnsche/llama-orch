// Hive SDK - TypeScript entry point
// Base URL: http://localhost:7835
//
// TEAM-374: Added HeartbeatMonitor for real-time worker updates

// ============================================================================
// TEAM-381: IMPORTANT - Types Should Come From Rust!
// ============================================================================
// 
// The following types are MANUALLY DEFINED but should be AUTO-GENERATED from Rust.
// 
// WHY? Single source of truth - types defined once in Rust, generated for TypeScript.
// HOW? Using `tsify` crate to auto-generate TypeScript from Rust structs.
// 
// TODO: Migrate these types to Rust (see bin/.plan/TEAM_381_HOW_TO_ADD_TYPES_FROM_RUST.md):
// - ProcessStats → bin/25_rbee_hive_crates/monitor/src/lib.rs (HAS Tsify!)
// - HiveInfo → bin/97_contracts/hive-contract/src/types.rs (HAS Tsify!)
// - HiveDevice → bin/97_contracts/hive-contract/src/heartbeat.rs (HAS Tsify!)
// 
// After migration:
// 1. Enable `wasm` feature in this SDK's Cargo.toml
// 2. Re-export types in lib.rs
// 3. Import from './pkg/bundler/rbee_hive_sdk'
// 4. Remove manual definitions below
// ============================================================================

// TEAM-374: ProcessStats structure (matches backend)
// TODO TEAM-381: Auto-generated from bin/25_rbee_hive_crates/monitor/src/lib.rs (Tsify ready!)
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

// TEAM-374: Hive info structure
// TODO TEAM-381: Auto-generated from bin/97_contracts/hive-contract/src/types.rs (Tsify ready!)
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

// TEAM-374: Hive heartbeat event (sent every 1s)
// TODO TEAM-381: This should be auto-generated from Rust
export interface HiveHeartbeatEvent {
  type: 'telemetry'
  hive_id: string
  hive_info: HiveInfo
  timestamp: string
  workers: ProcessStats[]
}

// TEAM-381: HuggingFace model types (for search - UI only, not from backend)
export interface HFModel {
  id: string
  modelId: string
  author: string
  downloads: number
  likes: number
  tags: string[]
  private: boolean
  gated: boolean | string
}

// Re-export WASM SDK types (including auto-generated TypeScript types from Rust)
export type { 
  HiveClient, 
  HeartbeatMonitor,
  OperationBuilder,
  ModelInfo, // TEAM-381: Auto-generated from Rust via tsify
} from './pkg/bundler/rbee_hive_sdk'
