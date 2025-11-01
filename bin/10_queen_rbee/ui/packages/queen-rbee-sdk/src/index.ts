// Queen-rbee SDK - TypeScript entry point
// Base URL: http://localhost:7833
//
// Queen UI Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor (future)
//
// Note: Worker/Model/Infer operations belong to Hive UI

// ============================================================================
// TEAM-381: ✅ ALL TYPES NOW AUTO-GENERATED FROM RUST!
// ============================================================================
// 
// HOW TO ADD NEW TYPES FROM RUST:
// 
// 1. Add type to contract crate (bin/97_contracts/hive-contract/src/telemetry.rs):
//    #[derive(Debug, Clone, Serialize, Deserialize)]
//    #[cfg_attr(feature = "wasm", derive(Tsify))]
//    #[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
//    pub struct YourType { ... }
//
// 2. Re-export from contract lib.rs:
//    pub use telemetry::YourType;
//
// 3. Re-export from SDK lib.rs:
//    pub use hive_contract::YourType;
//
// 4. Add dummy function in SDK types.rs:
//    #[wasm_bindgen]
//    pub fn __export_your_type(data: YourType) -> YourType { data }
//
// 5. Rebuild SDK: pnpm build
//
// 6. Import here from './pkg/bundler/queen_rbee_sdk'
//
// 7. DELETE any manual TypeScript definitions!
//
// See: bin/.plan/TEAM_381_HOW_TO_MIGRATE_TYPES_FROM_RUST.md
// ============================================================================


// TEAM-381: ✅ ALL TYPES AUTO-GENERATED FROM RUST! 
// Re-export WASM SDK types (all generated from Rust via tsify!)
import type { 
  QueenClient, 
  HeartbeatMonitor, 
  OperationBuilder,
  RhaiClient,
  RhaiScript,
  TestResult,
  // TEAM-381: ✅ ALL AUTO-GENERATED FROM RUST! 
  // These types are defined in hive-contract/telemetry.rs with Tsify annotations
  // Source: bin/97_contracts/hive-contract/src/telemetry.rs
  ProcessStats,
  HiveInfo,
  HiveDevice,
  HiveTelemetry,
  QueenHeartbeat,
  HeartbeatSnapshot,
} from '../pkg/bundler/queen_rbee_sdk'

export type {
  QueenClient, 
  HeartbeatMonitor, 
  OperationBuilder,
  RhaiClient,
  RhaiScript,
  TestResult,
  // TEAM-381: ✅ ALL AUTO-GENERATED FROM RUST! 
  // These types are defined in hive-contract/telemetry.rs with Tsify annotations
  // Source: bin/97_contracts/hive-contract/src/telemetry.rs
  ProcessStats,
  HiveInfo,
  HiveDevice,
  HiveTelemetry,
  QueenHeartbeat,
  HeartbeatSnapshot,
}

// TEAM-381: HeartbeatEvent union type (combines HiveTelemetry | QueenHeartbeat)
// This matches the Rust enum in bin/10_queen_rbee/src/http/heartbeat.rs
export type HeartbeatEvent = HiveTelemetry | QueenHeartbeat
