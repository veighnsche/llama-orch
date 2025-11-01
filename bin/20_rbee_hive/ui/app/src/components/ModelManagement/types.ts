// TEAM-381: Shared types for Model Management
// ModelInfo is auto-generated from Rust via tsify (single source of truth)
// HFModel is UI-only (HuggingFace API response)

// Re-export types from SDK
export type { ModelInfo, HFModel } from '@rbee/rbee-hive-react'

// UI-specific types
export type ViewMode = 'downloaded' | 'loaded' | 'search'

export interface FilterState {
  formats: string[]
  architectures: string[]
  maxSize: string
  openSourceOnly: boolean
  sortBy: 'downloads' | 'likes' | 'recent'
}
