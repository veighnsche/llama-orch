// Hive SDK - TypeScript entry point
// Base URL: http://localhost:7835
//
// TEAM-374: Added HeartbeatMonitor for real-time worker updates

// TEAM-374: ProcessStats structure (matches backend)
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
export interface HiveHeartbeatEvent {
  type: 'telemetry'
  hive_id: string
  hive_info: HiveInfo
  timestamp: string
  workers: ProcessStats[]
}

// Re-export WASM SDK types
export type { 
  HiveClient, 
  HeartbeatMonitor,
  OperationBuilder,
} from './pkg/bundler/rbee_hive_sdk'
