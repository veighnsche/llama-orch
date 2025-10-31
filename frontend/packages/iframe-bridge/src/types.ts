/**
 * TEAM-351: Generic iframe message types
 * TEAM-351: Bug fixes - Type safety, validation, extensibility
 */

/**
 * Valid message types
 * TEAM-351: Type-safe message types
 * TEAM-375: Added THEME_CHANGE for parent → iframe theme sync
 */
export type MessageType = 'NARRATION_EVENT' | 'COMMAND' | 'RESPONSE' | 'ERROR' | 'THEME_CHANGE'

/**
 * Base message interface
 * TEAM-351: All messages must extend this
 */
export interface BaseMessage {
  type: MessageType
  source: string
  timestamp: number
  id?: string          // Optional message ID for correlation
  version?: string     // Protocol version
}

/**
 * Narration event message
 * TEAM-351: Type-safe payload
 */
export interface NarrationMessage extends BaseMessage {
  type: 'NARRATION_EVENT'
  payload: {
    actor: string
    action: string
    human: string
    [key: string]: any  // Allow additional fields
  }
}

/**
 * Command message
 * TEAM-351: Type-safe args
 */
export interface CommandMessage extends BaseMessage {
  type: 'COMMAND'
  command: string
  args?: Record<string, unknown>  // Type-safe args object
}

/**
 * Response message
 * TEAM-351: For request/response pattern
 */
export interface ResponseMessage extends BaseMessage {
  type: 'RESPONSE'
  requestId: string
  success: boolean
  data?: unknown
  error?: string
}

/**
 * Error message
 * TEAM-351: For error reporting
 */
export interface ErrorMessage extends BaseMessage {
  type: 'ERROR'
  error: string
  code?: string
  details?: Record<string, unknown>
}

/**
 * Theme change message
 * TEAM-375: For parent → iframe theme synchronization
 */
export interface ThemeChangeMessage extends BaseMessage {
  type: 'THEME_CHANGE'
  theme: 'light' | 'dark'
}

/**
 * Union of all message types
 * TEAM-351: Type-safe message union
 * TEAM-375: Added ThemeChangeMessage
 */
export type IframeMessage = NarrationMessage | CommandMessage | ResponseMessage | ErrorMessage | ThemeChangeMessage

/**
 * Message validation result
 * TEAM-351: For validation feedback
 */
export interface ValidationResult {
  valid: boolean
  error?: string
  missing?: string[]
}

/**
 * Validate base message structure
 * TEAM-351: Runtime validation
 */
export function isValidBaseMessage(obj: any): obj is BaseMessage {
  return (
    obj !== null &&
    typeof obj === 'object' &&
    typeof obj.type === 'string' &&
    typeof obj.source === 'string' &&
    typeof obj.timestamp === 'number'
  )
}

/**
 * Validate iframe message
 * TEAM-351: Type guard with validation
 */
export function isValidIframeMessage(obj: any): obj is IframeMessage {
  if (!isValidBaseMessage(obj)) {
    return false
  }
  
  // TEAM-351: Type-specific validation
  const data = obj as any  // Use any for property access
  
  switch (data.type) {
    case 'NARRATION_EVENT':
      return !!(
        data.payload &&
        typeof data.payload === 'object' &&
        typeof data.payload.actor === 'string' &&
        typeof data.payload.action === 'string' &&
        typeof data.payload.human === 'string'
      )
    
    case 'COMMAND':
      return typeof data.command === 'string'
    
    case 'RESPONSE':
      return (
        typeof data.requestId === 'string' &&
        typeof data.success === 'boolean'
      )
    
    case 'ERROR':
      return typeof data.error === 'string'
    
    case 'THEME_CHANGE':
      return (
        typeof data.theme === 'string' &&
        (data.theme === 'light' || data.theme === 'dark')
      )
    
    default:
      return false
  }
}

/**
 * Validate message with detailed feedback
 * TEAM-351: Detailed validation for debugging
 */
export function validateMessage(obj: any): ValidationResult {
  if (!obj || typeof obj !== 'object') {
    return { valid: false, error: 'Message must be an object' }
  }
  
  const missing: string[] = []
  
  if (typeof obj.type !== 'string') missing.push('type')
  if (typeof obj.source !== 'string') missing.push('source')
  if (typeof obj.timestamp !== 'number') missing.push('timestamp')
  
  if (missing.length > 0) {
    return { valid: false, error: 'Missing required fields', missing }
  }
  
  if (!isValidIframeMessage(obj)) {
    return { valid: false, error: `Invalid message type or structure: ${obj.type}` }
  }
  
  return { valid: true }
}
