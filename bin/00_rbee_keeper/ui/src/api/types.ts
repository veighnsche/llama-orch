// TEAM-294: TypeScript types for Tauri command responses
// Mirrors the Rust CommandResponse struct from tauri_commands.rs

export interface CommandResponse {
  success: boolean
  message: string
  data?: string
}

// Request types for various operations
export interface HiveInstallRequest {
  host: string
  binary?: string
  install_dir?: string
}

export interface HiveUninstallRequest {
  host: string
  install_dir?: string
}

export interface HiveStartRequest {
  host: string
  install_dir?: string
  port: number
}

export interface WorkerSpawnRequest {
  hive_id: string
  model: string
  device: string
}

export interface ModelDownloadRequest {
  hive_id: string
  model: string
}

export interface InferRequest {
  hive_id: string
  model: string
  prompt: string
  max_tokens?: number
  temperature?: number
  top_p?: number
  top_k?: number
  device?: string
  worker_id?: string
  stream?: boolean
}
