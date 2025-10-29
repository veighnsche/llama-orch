// TEAM-291: React hooks for rbee WASM SDK

// TEAM-291: Re-export types from @rbee/queen-rbee-sdk for convenience
// TEAM-295: Fixed import to use @rbee/queen-rbee-sdk instead of @rbee/sdk
export type { HeartbeatMonitor, OperationBuilder, QueenClient } from '@rbee/queen-rbee-sdk'

// Export hooks
export { useRbeeSDK } from './hooks/useRbeeSDK'
export { useHeartbeat } from './hooks/useHeartbeat'
export type { HeartbeatData, UseHeartbeatResult } from './hooks/useHeartbeat'
export { useRhaiScripts } from './hooks/useRhaiScripts'
export type { RhaiScript, TestResult, UseRhaiScriptsResult } from './hooks/useRhaiScripts'

// TEAM-XXX: Export narration bridge utilities
export { sendNarrationToParent, parseNarrationLine, createNarrationStreamHandler } from './utils/narrationBridge'
export type { NarrationEvent, NarrationMessage } from './utils/narrationBridge'

export type { LoadOptions, RbeeSDK } from './types'
