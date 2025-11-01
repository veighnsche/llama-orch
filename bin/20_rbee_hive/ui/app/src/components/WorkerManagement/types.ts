// TEAM-382: Shared types for Worker Management
// ProcessStats is auto-generated from Rust via tsify (single source of truth)

// Re-export types from SDK
export type { ProcessStats } from '@rbee/rbee-hive-react'

// UI-specific types
export type ViewMode = 'catalog' | 'active' | 'spawn'

export interface SpawnFormState {
  modelId: string
  workerType: 'cpu' | 'cuda' | 'metal'
  deviceId: number
}
