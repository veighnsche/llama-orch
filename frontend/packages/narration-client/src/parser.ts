import type { BackendNarrationEvent, ParseStats } from './types'
import { isValidNarrationEvent } from './types'

/**
 * TEAM-351: Bug fixes - Validation, edge cases, monitoring
 */

// TEAM-351: Parse statistics for monitoring
let parseStats: ParseStats = {
  total: 0,
  success: 0,
  failed: 0,
  doneMarkers: 0,
  emptyLines: 0,
}

/**
 * Get current parse statistics
 * TEAM-351: Monitoring helper
 */
export function getParseStats(): Readonly<ParseStats> {
  return { ...parseStats }
}

/**
 * Reset parse statistics
 * TEAM-351: Monitoring helper
 */
export function resetParseStats(): void {
  parseStats = {
    total: 0,
    success: 0,
    failed: 0,
    doneMarkers: 0,
    emptyLines: 0,
  }
}

/**
 * Parse SSE line into narration event
 * 
 * TEAM-351: Bug fixes - Validation, empty strings, SSE format, error handling
 * 
 * @param line - Raw SSE line from backend
 * @param options - Parse options
 * @returns Parsed event or null if invalid/skipped
 */
export function parseNarrationLine(
  line: string,
  options: { silent?: boolean; validate?: boolean } = {}
): BackendNarrationEvent | null {
  const { silent = false, validate = true } = options
  
  parseStats.total++
  
  // TEAM-351: Handle empty strings and whitespace-only lines
  if (!line || line.trim().length === 0) {
    parseStats.emptyLines++
    return null
  }
  
  // TEAM-351: Skip [DONE] marker gracefully (not an error)
  const trimmed = line.trim()
  if (trimmed === '[DONE]') {
    parseStats.doneMarkers++
    return null
  }
  
  // TEAM-351: Skip SSE comment lines (start with ':')
  if (trimmed.startsWith(':')) {
    return null
  }
  
  // TEAM-351: Skip SSE event: and id: lines (we only care about data:)
  if (trimmed.startsWith('event:') || trimmed.startsWith('id:')) {
    return null
  }
  
  try {
    // TEAM-351: Remove SSE "data: " prefix if present
    let jsonStr = trimmed
    if (jsonStr.startsWith('data:')) {
      jsonStr = jsonStr.slice(5).trim()
    }
    
    // TEAM-351: Handle empty data after prefix removal
    if (jsonStr.length === 0) {
      parseStats.emptyLines++
      return null
    }
    
    // TEAM-351: Parse JSON with error handling
    const event = JSON.parse(jsonStr)
    
    // TEAM-351: Validate event structure if requested
    if (validate && !isValidNarrationEvent(event)) {
      parseStats.failed++
      if (!silent) {
        console.warn('[NarrationClient] Invalid event structure:', {
          line: line.substring(0, 100),
          event,
          missing: {
            actor: !event?.actor,
            action: !event?.action,
            human: !event?.human,
          },
        })
      }
      return null
    }
    
    parseStats.success++
    return event as BackendNarrationEvent
  } catch (error) {
    parseStats.failed++
    if (!silent) {
      console.warn('[NarrationClient] Failed to parse line:', {
        line: line.substring(0, 100),
        error: error instanceof Error ? error.message : String(error),
      })
    }
    return null
  }
}
