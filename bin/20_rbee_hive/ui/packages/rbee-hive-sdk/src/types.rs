//! Type exports for TypeScript generation
//!
//! TEAM-381: This module exists solely to force Tsify types to be generated in the .d.ts file
//! By using the types in WASM functions, they get included in the TypeScript definitions

use hive_contract::{HiveHeartbeatEvent, HiveInfo, ProcessStats};
use wasm_bindgen::prelude::*;

/// TEAM-381: Dummy function to force ProcessStats into TypeScript definitions
/// This function will never be called - it exists only to make wasm-bindgen generate the type
#[wasm_bindgen]
pub fn __export_process_stats_type(data: ProcessStats) -> ProcessStats {
    data
}

/// TEAM-381: Dummy function to force HiveInfo into TypeScript definitions
/// This function will never be called - it exists only to make wasm-bindgen generate the type
#[wasm_bindgen]
pub fn __export_hive_info_type(data: HiveInfo) -> HiveInfo {
    data
}

/// TEAM-381: Dummy function to force HiveHeartbeatEvent into TypeScript definitions
/// This function will never be called - it exists only to make wasm-bindgen generate the type
#[wasm_bindgen]
pub fn __export_hive_heartbeat_event_type(data: HiveHeartbeatEvent) -> HiveHeartbeatEvent {
    data
}
