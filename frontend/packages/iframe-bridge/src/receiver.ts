import type { IframeMessage } from './types'
import { isValidIframeMessage, validateMessage } from './types'
import { createOriginValidator, type OriginConfig } from './validator'

/**
 * TEAM-351: Message receiver for parent ← iframe communication
 * TEAM-351: Bug fixes - Validation, error handling, monitoring, memory leak prevention
 */

export interface ReceiverConfig extends OriginConfig {
  onMessage: (message: IframeMessage) => void
  onError?: (error: Error, message?: any) => void
  debug?: boolean
  validate?: boolean
}

/**
 * Receive statistics for monitoring
 * TEAM-351: Track receive success/failure
 */
export interface ReceiveStats {
  total: number
  accepted: number
  rejected: number
  invalidOrigin: number
  invalidMessage: number
  errors: number
}

// TEAM-351: Production mode detection
const isProduction = (import.meta as any).env?.PROD ?? false

// TEAM-351: Receive statistics
let receiveStats: ReceiveStats = {
  total: 0,
  accepted: 0,
  rejected: 0,
  invalidOrigin: 0,
  invalidMessage: 0,
  errors: 0,
}

/**
 * Get receive statistics
 * TEAM-351: Monitoring helper
 */
export function getReceiveStats(): Readonly<ReceiveStats> {
  return { ...receiveStats }
}

/**
 * Reset receive statistics
 * TEAM-351: Monitoring helper
 */
export function resetReceiveStats(): void {
  receiveStats = {
    total: 0,
    accepted: 0,
    rejected: 0,
    invalidOrigin: 0,
    invalidMessage: 0,
    errors: 0,
  }
}

/**
 * Active receivers tracking
 * TEAM-351: Prevent memory leaks
 */
const activeReceivers = new Set<() => void>()

/**
 * Get count of active receivers
 * TEAM-351: Memory leak detection
 */
export function getActiveReceiverCount(): number {
  return activeReceivers.size
}

/**
 * Cleanup all active receivers
 * TEAM-351: Emergency cleanup
 */
export function cleanupAllReceivers(): void {
  activeReceivers.forEach(cleanup => cleanup())
  activeReceivers.clear()
}

/**
 * Create message receiver with validation and error handling
 * 
 * TEAM-351: Bug fixes - Validation, error handling, memory leak prevention
 * 
 * @param config - Receiver configuration
 * @returns Cleanup function
 * @throws Error if config is invalid
 */
export function createMessageReceiver(config: ReceiverConfig) {
  const { 
    onMessage, 
    onError,
    debug = !isProduction, 
    validate = true,
  } = config
  
  // TEAM-351: Validate config
  if (typeof onMessage !== 'function') {
    throw new Error('Invalid receiver config: onMessage must be a function')
  }
  
  const validateOrigin = createOriginValidator(config)
  
  const handleMessage = (event: MessageEvent) => {
    receiveStats.total++
    
    // TEAM-351: Validate origin
    if (!validateOrigin(event.origin)) {
      receiveStats.invalidOrigin++
      if (debug) {
        console.warn('[IframeBridge] Rejected origin:', {
          origin: event.origin,
          allowed: config.allowedOrigins,
        })
      }
      return
    }
    
    // TEAM-351: Validate message structure
    if (!event.data || typeof event.data !== 'object' || !event.data.type) {
      receiveStats.invalidMessage++
      if (debug) {
        console.warn('[IframeBridge] Invalid message structure:', event.data)
      }
      return
    }
    
    // TEAM-351: Validate message if requested
    if (validate) {
      const validation = validateMessage(event.data)
      if (!validation.valid) {
        receiveStats.invalidMessage++
        if (debug) {
          console.warn('[IframeBridge] Message validation failed:', {
            error: validation.error,
            missing: validation.missing,
          })
        }
        return
      }
    }
    
    // TEAM-351: Call onMessage with error handling
    try {
      receiveStats.accepted++
      onMessage(event.data as IframeMessage)
    } catch (error) {
      receiveStats.errors++
      console.error('[IframeBridge] onMessage handler error:', error)
      
      if (onError) {
        try {
          onError(
            error instanceof Error ? error : new Error(String(error)),
            event.data
          )
        } catch (onErrorError) {
          console.error('[IframeBridge] onError handler error:', onErrorError)
        }
      }
    }
  }

  window.addEventListener('message', handleMessage)
  
  // TEAM-351: Cleanup function with memory leak prevention
  const cleanup = () => {
    window.removeEventListener('message', handleMessage)
    activeReceivers.delete(cleanup)
  }
  
  // TEAM-351: Track active receiver
  activeReceivers.add(cleanup)
  
  return cleanup
}
