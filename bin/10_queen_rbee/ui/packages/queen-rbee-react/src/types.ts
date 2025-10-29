// TEAM-291: Type definitions for rbee SDK
// TEAM-295: Fixed import to use @rbee/queen-rbee-sdk instead of @rbee/sdk

import type { HeartbeatMonitor, OperationBuilder, QueenClient, RhaiClient } from '@rbee/queen-rbee-sdk'

export interface RbeeSDK {
  QueenClient: typeof QueenClient
  HeartbeatMonitor: typeof HeartbeatMonitor
  OperationBuilder: typeof OperationBuilder
  RhaiClient: typeof RhaiClient
}

export type LoadOptions = {
  timeoutMs?: number
  maxAttempts?: number
  baseBackoffMs?: number
  initArg?: unknown
  onReady?: (sdk: RbeeSDK) => void
}

export type GlobalSlot = {
  promise?: Promise<{ sdk: RbeeSDK }>
  value?: { sdk: RbeeSDK }
  error?: Error
}
