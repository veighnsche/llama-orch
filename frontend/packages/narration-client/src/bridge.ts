import type { BackendNarrationEvent, NarrationMessage } from './types'
import type { ServiceConfig } from './config'
import { getParentOrigin, isValidServiceConfig } from './config'
import { parseNarrationLine } from './parser'

/**
 * TEAM-351: Bug fixes - Validation, production logging, error handling, retry logic
 */

// TEAM-351: Protocol version for future compatibility
const PROTOCOL_VERSION = '1.0.0'

// TEAM-351: Production mode detection (browser-safe)
const isProduction = (import.meta as any).env?.PROD ?? false

/**
 * Send narration event to parent window (rbee-keeper)
 * 
 * TEAM-351: Bug fixes - Validation, production logging, error handling
 * 
 * @param event - Narration event to send
 * @param serviceConfig - Service configuration
 * @param options - Send options
 * @returns true if sent successfully, false otherwise
 */
export function sendToParent(
  event: BackendNarrationEvent,
  serviceConfig: ServiceConfig,
  options: { debug?: boolean; retry?: boolean } = {}
): boolean {
  const { debug = !isProduction, retry = false } = options
  
  // TEAM-351: Validate environment
  if (typeof window === 'undefined' || window.parent === window) {
    return false
  }
  
  // TEAM-351: Validate service config
  if (!isValidServiceConfig(serviceConfig)) {
    console.error('[NarrationClient] Invalid service config:', serviceConfig)
    return false
  }

  const message: NarrationMessage = {
    type: 'NARRATION_EVENT',
    payload: event,
    source: serviceConfig.name,
    timestamp: Date.now(),
    version: PROTOCOL_VERSION,
  }

  try {
    const parentOrigin = getParentOrigin(serviceConfig)
    
    // TEAM-351: Only log in debug mode (not production)
    if (debug) {
      console.log(`[${serviceConfig.name}] Sending to parent:`, {
        origin: parentOrigin,
        action: event.action,
        actor: event.actor,
      })
    }
    
    window.parent.postMessage(message, parentOrigin)
    return true
  } catch (error) {
    console.warn(`[${serviceConfig.name}] Failed to send to parent:`, {
      error: error instanceof Error ? error.message : String(error),
      action: event.action,
    })
    
    // TEAM-351: Retry once if requested
    if (retry) {
      try {
        setTimeout(() => {
          window.parent.postMessage(message, getParentOrigin(serviceConfig))
        }, 100)
      } catch (retryError) {
        console.error(`[${serviceConfig.name}] Retry failed:`, retryError)
      }
    }
    
    return false
  }
}

/**
 * Create a stream handler for SSE narration events
 * 
 * TEAM-351: Bug fixes - Options, error handling, validation
 * 
 * @param serviceConfig - Service configuration
 * @param onLocal - Optional local handler for events
 * @param options - Handler options
 * @returns Stream handler function
 */
export function createStreamHandler(
  serviceConfig: ServiceConfig,
  onLocal?: (event: BackendNarrationEvent) => void,
  options: {
    debug?: boolean
    silent?: boolean
    validate?: boolean
    retry?: boolean
  } = {}
) {
  const { debug = !isProduction, silent = false, validate = true, retry = false } = options
  
  // TEAM-351: Validate service config at creation time
  if (!isValidServiceConfig(serviceConfig)) {
    throw new Error(`Invalid service config: ${JSON.stringify(serviceConfig)}`)
  }
  
  return (line: string) => {
    const event = parseNarrationLine(line, { silent, validate })
    if (event) {
      sendToParent(event, serviceConfig, { debug, retry })
      
      // TEAM-351: Error handling for local handler
      if (onLocal) {
        try {
          onLocal(event)
        } catch (error) {
          console.error(`[${serviceConfig.name}] Local handler error:`, error)
        }
      }
    }
  }
}
