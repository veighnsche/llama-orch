// TEAM-XXX: Narration Event Bridge for iframe → parent communication
// Sends narration events from Queen UI (iframe) to rbee-keeper (parent)

export interface NarrationEvent {
  actor: string
  action: string
  human: string
  level?: string
  timestamp?: number
  job_id?: string
}

export interface NarrationMessage {
  type: 'QUEEN_NARRATION'
  payload: NarrationEvent
  source: 'queen-rbee'
  timestamp: number
}

/**
 * Send narration event to parent window (rbee-keeper)
 * 
 * This bridges the iframe boundary so narration events from Queen
 * can appear in rbee-keeper's narration panel.
 */
export function sendNarrationToParent(event: NarrationEvent): void {
  // Only send if we're in an iframe
  if (typeof window === 'undefined' || window.parent === window) {
    return
  }

  const message: NarrationMessage = {
    type: 'QUEEN_NARRATION',
    payload: event,
    source: 'queen-rbee',
    timestamp: Date.now(),
  }

  try {
    // Send to parent (rbee-keeper at localhost:7834)
    window.parent.postMessage(message, 'http://localhost:7834')
  } catch (error) {
    console.warn('[Queen] Failed to send narration to parent:', error)
  }
}

/**
 * Parse SSE narration line into NarrationEvent
 * 
 * Expected format: "data: {json}\n\n"
 */
export function parseNarrationLine(line: string): NarrationEvent | null {
  try {
    // Remove "data: " prefix if present
    const jsonStr = line.startsWith('data: ') ? line.slice(6) : line
    
    // Parse JSON
    const event = JSON.parse(jsonStr.trim())
    
    // Validate structure
    if (event && typeof event === 'object' && event.actor && event.action) {
      return event as NarrationEvent
    }
    
    return null
  } catch (error) {
    console.warn('[Queen] Failed to parse narration line:', line, error)
    return null
  }
}

/**
 * Stream handler that sends narration events to parent
 */
export function createNarrationStreamHandler(onLocal?: (event: NarrationEvent) => void) {
  return (line: string) => {
    const event = parseNarrationLine(line)
    if (event) {
      // Send to parent window
      sendNarrationToParent(event)
      
      // Also handle locally if callback provided
      if (onLocal) {
        onLocal(event)
      }
    }
  }
}
