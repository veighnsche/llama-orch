// TEAM-291: React hooks for rbee WASM SDK
// TEAM-352: SDK loader migrated to @rbee/sdk-loader package
// TEAM-352: Hooks now import directly from @rbee/sdk-loader - NO WRAPPERS
// TEAM-377: React Query removed - use @rbee/ui/providers instead

// TEAM-377: React Query REMOVED
// DO NOT re-export React Query - import from @rbee/ui/providers:
//   import { QueryProvider } from '@rbee/ui/providers'
// This ensures consistent configuration across all apps

// TEAM-291: Re-export types from @rbee/queen-rbee-sdk for convenience
// TEAM-295: Fixed import to use @rbee/queen-rbee-sdk instead of @rbee/sdk
export type { HeartbeatMonitor, OperationBuilder, QueenClient } from '@rbee/queen-rbee-sdk'

// Export hooks
// TEAM-377: RULE ZERO - Deleted useRbeeSDK alias, only export useQueenSDK
export { useQueenSDK } from './hooks/useQueenSDK'
export { useHeartbeat } from './hooks/useHeartbeat'
export type { HeartbeatData, UseHeartbeatResult } from './hooks/useHeartbeat'
export { useRhaiScripts } from './hooks/useRhaiScripts'
export type { RhaiScript, TestResult, UseRhaiScriptsResult } from './hooks/useRhaiScripts'

// TEAM-352: Narration bridge DELETED - RULE ZERO compliance
// DO NOT re-export wrappers - import directly from @rbee/narration-client:
//   import { createStreamHandler, SERVICES } from '@rbee/narration-client'
//   import type { BackendNarrationEvent } from '@rbee/narration-client'

// TEAM-352: Export RbeeSDK type (LoadOptions removed - use @rbee/sdk-loader directly)
export type { RbeeSDK } from './types'

// NOTE: loader.ts is intentionally NOT exported
// Hooks import directly from @rbee/sdk-loader
