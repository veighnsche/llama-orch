/**
 * Narration event from backend SSE stream
 * This is the format ALL backends send (Queen, Hive, Worker)
 * 
 * TEAM-351: Shared narration types
 * TEAM-351: Bug fixes - Validation, type safety
 */
export interface BackendNarrationEvent {
  actor: string          // REQUIRED: Service name (queen, hive, worker)
  action: string         // REQUIRED: Action being performed
  human: string          // REQUIRED: Human-readable message
  formatted?: string     // Optional: Contains function name with ANSI codes
  level?: string         // Optional: Log level (info, warn, error)
  timestamp?: number     // Optional: Unix timestamp
  job_id?: string        // Optional: Job ID for SSE routing
  target?: string        // Optional: Target service
  correlation_id?: string // Optional: End-to-end tracing ID
}

/**
 * Validate that an object is a valid BackendNarrationEvent
 * TEAM-351: Runtime validation for type safety
 */
export function isValidNarrationEvent(obj: any): obj is BackendNarrationEvent {
  return (
    obj !== null &&
    typeof obj === 'object' &&
    typeof obj.actor === 'string' && obj.actor.length > 0 &&
    typeof obj.action === 'string' && obj.action.length > 0 &&
    typeof obj.human === 'string' && obj.human.length > 0
  )
}

/**
 * Valid service names for narration
 * TEAM-351: Type-safe service names
 */
export type ServiceName = 'queen' | 'hive' | 'worker'

/**
 * Message sent to parent window via postMessage
 * TEAM-351: Type-safe message format
 */
export interface NarrationMessage {
  type: 'NARRATION_EVENT'
  payload: BackendNarrationEvent
  source: string           // Service name (queen-rbee, rbee-hive, llm-worker)
  timestamp: number
  version: string          // Protocol version for future compatibility
}

/**
 * Parse statistics for monitoring
 * TEAM-351: Track parsing success/failure
 */
export interface ParseStats {
  total: number
  success: number
  failed: number
  doneMarkers: number
  emptyLines: number
}
