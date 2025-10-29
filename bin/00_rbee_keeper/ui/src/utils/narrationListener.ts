// TEAM-XXX: Listen for narration events from Queen iframe
// Receives postMessage events and adds them to narration store

import { useNarrationStore } from '../store/narrationStore'
import type { NarrationEvent } from '../generated/bindings'

export interface NarrationMessage {
  type: 'QUEEN_NARRATION'
  payload: NarrationEvent
  source: 'queen-rbee'
  timestamp: number
}

/**
 * Setup listener for narration events from Queen iframe
 * Call this once at app startup
 */
export function setupNarrationListener(): () => void {
  const handleMessage = (event: MessageEvent) => {
    // Security: Verify origin is Queen
    if (event.origin !== 'http://localhost:7833') {
      return
    }

    // Filter for Queen narration events
    if (event.data?.type === 'QUEEN_NARRATION') {
      const message = event.data as NarrationMessage
      
      console.log('[Keeper] Received narration from Queen:', message.payload)
      
      // Add to narration store
      useNarrationStore.getState().addEntry(message.payload)
    }
  }

  // Add listener
  window.addEventListener('message', handleMessage)

  // Return cleanup function
  return () => {
    window.removeEventListener('message', handleMessage)
  }
}
