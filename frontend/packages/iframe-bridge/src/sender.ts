import type { IframeMessage } from './types'
import { isValidIframeMessage, validateMessage } from './types'
import { isValidOriginFormat } from './validator'

/**
 * TEAM-351: Message sender for iframe → parent communication
 * TEAM-351: Bug fixes - Validation, return values, production logging, retry
 */

export interface SenderConfig {
  targetOrigin: string
  debug?: boolean
  validate?: boolean
  timeout?: number      // Timeout for send operation (ms)
  retry?: boolean       // Retry once on failure
}

/**
 * Send statistics for monitoring
 * TEAM-351: Track send success/failure
 */
export interface SendStats {
  total: number
  success: number
  failed: number
  retried: number
}

// TEAM-351: Production mode detection
const isProduction = (import.meta as any).env?.PROD ?? false

// TEAM-351: Send statistics
let sendStats: SendStats = {
  total: 0,
  success: 0,
  failed: 0,
  retried: 0,
}

/**
 * Get send statistics
 * TEAM-351: Monitoring helper
 */
export function getSendStats(): Readonly<SendStats> {
  return { ...sendStats }
}

/**
 * Reset send statistics
 * TEAM-351: Monitoring helper
 */
export function resetSendStats(): void {
  sendStats = {
    total: 0,
    success: 0,
    failed: 0,
    retried: 0,
  }
}

/**
 * Validate sender config
 * TEAM-351: Config validation
 */
export function isValidSenderConfig(config: any): config is SenderConfig {
  return (
    config !== null &&
    typeof config === 'object' &&
    typeof config.targetOrigin === 'string' &&
    isValidOriginFormat(config.targetOrigin)
  )
}

/**
 * Create message sender with validation and monitoring
 * 
 * TEAM-351: Bug fixes - Validation, return values, production logging
 * 
 * @param config - Sender configuration
 * @returns Sender function that returns boolean (success/failure)
 * @throws Error if config is invalid
 */
export function createMessageSender(config: SenderConfig) {
  const { 
    targetOrigin, 
    debug = !isProduction, 
    validate = true,
    timeout = 5000,
    retry = false,
  } = config
  
  // TEAM-351: Validate config at creation time
  if (!isValidSenderConfig(config)) {
    throw new Error(`Invalid sender config: targetOrigin must be a valid origin URL`)
  }
  
  return (message: IframeMessage): boolean => {
    sendStats.total++
    
    // TEAM-351: Validate environment
    if (typeof window === 'undefined' || window.parent === window) {
      return false
    }
    
    // TEAM-351: Validate message if requested
    if (validate) {
      const validation = validateMessage(message)
      if (!validation.valid) {
        console.error('[IframeBridge] Invalid message:', validation.error, validation.missing)
        sendStats.failed++
        return false
      }
    }

    try {
      // TEAM-351: Only log in debug mode
      if (debug) {
        console.log('[IframeBridge] Sending:', {
          type: message.type,
          source: message.source,
          target: targetOrigin,
        })
      }
      
      window.parent.postMessage(message, targetOrigin)
      sendStats.success++
      return true
    } catch (error) {
      console.warn('[IframeBridge] Send failed:', {
        error: error instanceof Error ? error.message : String(error),
        type: message.type,
        target: targetOrigin,
      })
      
      // TEAM-351: Retry once if requested
      if (retry) {
        try {
          setTimeout(() => {
            try {
              window.parent.postMessage(message, targetOrigin)
              sendStats.retried++
            } catch (retryError) {
              console.error('[IframeBridge] Retry failed:', retryError)
            }
          }, 100)
        } catch (retryError) {
          // Ignore retry errors
        }
      }
      
      sendStats.failed++
      return false
    }
  }
}
