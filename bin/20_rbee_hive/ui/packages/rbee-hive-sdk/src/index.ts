// Hive SDK - TypeScript entry point
// Base URL: http://localhost:7835
//
// TEAM-374: Added HeartbeatMonitor for real-time worker updates
// TEAM-381: ✅ ALL TYPES AUTO-GENERATED FROM RUST!

// ============================================================================
// TEAM-381: ✅ MIGRATION COMPLETE!
// ============================================================================
// 
// All types are now auto-generated from Rust contract crates:
// - ProcessStats: bin/97_contracts/hive-contract/src/telemetry.rs
// - HiveInfo: bin/97_contracts/hive-contract/src/types.rs
// - HiveHeartbeatEvent: bin/97_contracts/hive-contract/src/telemetry.rs
// 
// Single source of truth: RUST! 🦀
// No more manual TypeScript type definitions!
// 
// See: bin/.plan/TEAM_381_HOW_TO_MIGRATE_TYPES_FROM_RUST.md
// ============================================================================

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
  // TEAM-381: ✅ ALL AUTO-GENERATED FROM RUST!
  // Source: bin/97_contracts/hive-contract/src/telemetry.rs
  ProcessStats,
  HiveHeartbeatEvent,
  // Source: bin/97_contracts/hive-contract/src/types.rs
  HiveInfo,
  // Source: bin/97_contracts/operations-contract/
  ModelInfo,
} from '../pkg/bundler/rbee_hive_sdk'
