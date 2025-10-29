/**
 * TEAM-351: Simplified SSE parser using eventsource-parser library
 * 
 * TEAM-356: Replaced ~80 LOC of custom SSE parsing with battle-tested library
 * - Uses eventsource-parser (3.6M weekly downloads, 0 dependencies, 2.4kb gzipped)
 * - Handles SSE format (data:, event:, id:, retry:, comments)
 * - Handles multi-line events
 * - Streaming parser (no buffering)
 * 
 * Kept custom:
 * - Event validation (project-specific)
 * - Parse statistics (monitoring)
 * - [DONE] marker handling
 */

import { createParser, type EventSourceMessage } from 'eventsource-parser'
import type { BackendNarrationEvent, ParseStats } from './types'
import { isValidNarrationEvent } from './types'

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
 * Create a streaming SSE parser for narration events
 * 
 * TEAM-356: Uses eventsource-parser for robust SSE handling
 * 
 * @param onEvent - Callback for valid narration events
 * @param options - Parser options
 * @returns Parser with feed() method to process SSE chunks
 */
export function createNarrationParser(
  onEvent: (event: BackendNarrationEvent) => void,
  options: { 
    silent?: boolean
    validate?: boolean
    onError?: (error: Error, data?: string) => void
  } = {}
) {
  const { silent = false, validate = true, onError } = options
  
  const parser = createParser({
    onEvent: (message: EventSourceMessage) => {
      parseStats.total++
      
      const data = message.data
      
      // TEAM-351: Check for [DONE] marker
      if (data.trim() === '[DONE]') {
        parseStats.doneMarkers++
        return
      }
      
      // TEAM-351: Check for empty data
      if (!data || data.trim().length === 0) {
        parseStats.emptyLines++
        return
      }
      
      try {
        // Parse JSON event
        const event = JSON.parse(data)
        
        // TEAM-351: Validate event structure if requested
        if (validate && !isValidNarrationEvent(event)) {
          parseStats.failed++
          if (!silent) {
            console.warn('[NarrationClient] Invalid event structure:', {
              data: data.substring(0, 100),
              event,
              missing: {
                actor: !event?.actor,
                action: !event?.action,
                human: !event?.human,
              },
            })
          }
          return
        }
        
        parseStats.success++
        onEvent(event as BackendNarrationEvent)
      } catch (error) {
        parseStats.failed++
        const err = error instanceof Error ? error : new Error(String(error))
        
        if (!silent) {
          console.warn('[NarrationClient] Failed to parse event:', {
            data: data.substring(0, 100),
            error: err.message,
          })
        }
        
        if (onError) {
          onError(err, data)
        }
      }
    },
  })
  
  return {
    /**
     * Feed a chunk of SSE data to the parser
     * Can be a single line or multiple lines
     */
    feed: (chunk: string) => parser.feed(chunk),
    
    /**
     * Reset parser state between reconnections
     */
    reset: () => parser.reset(),
  }
}

/**
 * Parse a single SSE line (legacy compatibility)
 * 
 * TEAM-351: Original line-by-line parser
 * TEAM-356: For streaming use, prefer createNarrationParser() instead
 * 
 * @param line - Raw SSE line from backend
 * @param options - Parse options
 * @returns Parsed event or null if invalid/skipped
 */
export function parseNarrationLine(
  line: string,
  options: { silent?: boolean; validate?: boolean } = {}
): BackendNarrationEvent | null {
  let result: BackendNarrationEvent | null = null
  
  const parser = createNarrationParser(
    (event) => { result = event },
    options
  )
  
  // Add data: prefix if not present, then add double newline to signal end of SSE event
  const formatted = line.startsWith('data:') ? line : `data: ${line}`
  parser.feed(formatted + '\n\n')
  
  return result
}
