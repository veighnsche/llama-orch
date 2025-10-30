// Queen-rbee SDK - TypeScript entry point
// Base URL: http://localhost:7833
//
// Queen UI Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor (future)
//
// Note: Worker/Model/Infer operations belong to Hive UI

// TEAM-364: Updated to match backend ProcessStats structure
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

// TEAM-364: Hive telemetry event (sent every 1s from each hive)
export interface HiveTelemetry {
  type: 'hive_telemetry'
  hive_id: string
  timestamp: string
  workers: ProcessStats[]
}

// TEAM-364: Queen heartbeat event (sent every 2.5s)
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

// TEAM-364: Union type for all heartbeat events
export type HeartbeatEvent = HiveTelemetry | QueenHeartbeat

// Legacy interface (deprecated, use HeartbeatEvent instead)
export interface HeartbeatSnapshot {
  workers_online: number
  hives_online: number
  timestamp: string
  workers: ProcessStats[]
}

// Re-export WASM SDK types
export type { 
  QueenClient, 
  HeartbeatMonitor, 
  OperationBuilder,
  RhaiClient,
  RhaiScript,
  TestResult,
} from './pkg/bundler/rbee_sdk'
