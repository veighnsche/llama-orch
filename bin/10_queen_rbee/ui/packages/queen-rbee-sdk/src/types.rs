//! Type exports for TypeScript generation
//!
//! TEAM-381: This module exists solely to force Tsify types to be generated in the .d.ts file
//! By using the types in WASM functions, they get included in the TypeScript definitions

use hive_contract::{
    HeartbeatSnapshot, HiveDevice, HiveInfo, HiveTelemetry, ProcessStats, QueenHeartbeat,
};
use wasm_bindgen::prelude::*;

/// TEAM-381: Dummy function to force ProcessStats into TypeScript definitions
/// This function will never be called - it exists only to make wasm-bindgen generate the type
#[wasm_bindgen]
pub fn __export_process_stats_type(stats: ProcessStats) -> ProcessStats {
    stats
}

/// TEAM-381: Dummy function to force HiveInfo into TypeScript definitions
#[wasm_bindgen]
pub fn __export_hive_info_type(info: HiveInfo) -> HiveInfo {
    info
}

/// TEAM-381: Dummy function to force HiveDevice into TypeScript definitions
#[wasm_bindgen]
pub fn __export_hive_device_type(device: HiveDevice) -> HiveDevice {
    device
}

/// TEAM-381: Dummy function to force HiveTelemetry into TypeScript definitions
#[wasm_bindgen]
pub fn __export_hive_telemetry_type(telemetry: HiveTelemetry) -> HiveTelemetry {
    telemetry
}

/// TEAM-381: Dummy function to force QueenHeartbeat into TypeScript definitions
#[wasm_bindgen]
pub fn __export_queen_heartbeat_type(heartbeat: QueenHeartbeat) -> QueenHeartbeat {
    heartbeat
}

/// TEAM-381: Dummy function to force HeartbeatSnapshot into TypeScript definitions
#[wasm_bindgen]
pub fn __export_heartbeat_snapshot_type(snapshot: HeartbeatSnapshot) -> HeartbeatSnapshot {
    snapshot
}
