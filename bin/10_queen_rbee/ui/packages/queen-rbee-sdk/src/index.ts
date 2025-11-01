// Queen-rbee SDK - TypeScript entry point
// Base URL: http://localhost:7833
//
// Queen UI Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor (future)
//
// Note: Worker/Model/Infer operations belong to Hive UI

// ============================================================================
// TEAM-381: IMPORTANT - Types Should Come From Rust!
// ============================================================================
// 
// The following types are MANUALLY DEFINED but should be AUTO-GENERATED from Rust.
// 
// WHY? Single source of truth - types defined once in Rust, generated for TypeScript.
// HOW? Using `tsify` crate to auto-generate TypeScript from Rust structs.
// 
// TODO: Migrate these types to Rust:
// 1. Add `#[cfg_attr(feature = "wasm", derive(Tsify))]` to Rust struct
// 2. Enable `wasm` feature in SDK Cargo.toml
// 3. Re-export in SDK lib.rs
// 4. Re-export in this file from './pkg/bundler/rbee_sdk'
// 
// See: bin/.plan/TEAM_381_HOW_TO_ADD_TYPES_FROM_RUST.md
// ============================================================================


// Re-export WASM SDK types (including Rust-generated types via tsify)
export type { 
  QueenClient, 
  HeartbeatMonitor, 
  OperationBuilder,
  RhaiClient,
  RhaiScript,
  TestResult,
  // TEAM-381: Auto-generated from Rust via tsify
  ProcessStats,
  HiveInfo,
  HiveDevice,
} from './pkg/bundler/rbee_sdk'
