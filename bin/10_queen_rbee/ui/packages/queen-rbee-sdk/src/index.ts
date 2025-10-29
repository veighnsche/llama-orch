// Queen-rbee SDK - TypeScript entry point
// Base URL: http://localhost:7833
//
// Queen UI Purpose:
// 1. Heartbeat Monitor - Real-time worker/hive status
// 2. RHAI IDE - Scheduling script editor (future)
//
// Note: Worker/Model/Infer operations belong to Hive UI

export interface HeartbeatSnapshot {
  workers_online: number
  hives_online: number
  timestamp: string
  workers: WorkerInfo[]
}

export interface WorkerInfo {
  id: string
  model_id: string
  device: number
  port: number
  status: string
  last_heartbeat: string
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
