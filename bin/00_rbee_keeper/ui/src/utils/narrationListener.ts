// TEAM-352: FIXED - Listen for narration events from service iframes
// Uses @rbee/narration-bridge for proper message validation
// Uses @rbee/narration-client types for type safety
// Uses @rbee/shared-config for port configuration
// TEAM-413: Added download progress tracking from narration events

import { createMessageReceiver } from '@rbee/narration-bridge'
import type { BackendNarrationEvent } from '@rbee/narration-client'
import { getAllowedOrigins } from '@rbee/shared-config'
import type { NarrationEvent } from '../generated/bindings'
import { useDownloadStore } from '../store/downloadStore'
import { useNarrationStore } from '../store/narrationStore'

/**
 * Setup listener for narration events from service iframes (Queen, Hive, Worker)
 * Call this once at app startup
 *
 * TEAM-352: Fixed to use shared packages and correct message type
 */
export function setupNarrationListener(): () => void {
  // TEAM-352: Use shared-config for allowed origins (no hardcoded URLs)
  return createMessageReceiver({
    allowedOrigins: getAllowedOrigins(),
    onMessage: (message) => {
      // TEAM-352: Filter for NARRATION_EVENT type (from @rbee/narration-client)
      if (message.type === 'NARRATION_EVENT') {
        const narrationEvent = message.payload as BackendNarrationEvent

        console.log('[Keeper] Received narration:', narrationEvent)

        // TEAM-352: Extract function name from formatted field (contains ANSI codes)
        // Format: "\x1b[1mfunction_name\x1b[0m \x1b[2maction\x1b[0m\nmessage"
        // The \x1b[1m...\x1b[0m is the function name in bold
        const extractFnName = (formatted?: string): string | null => {
          if (!formatted) return null

          // Match text between ESC[1m (bold) and ESC[0m (reset)
          const match = formatted.match(/\x1b\[1m([^\x1b]+)\x1b\[0m/)
          return match?.[1] ? match[1] : null
        }

        // TEAM-352: Map shared narration format to Keeper's format
        const keeperEvent: NarrationEvent = {
          level: narrationEvent.level || 'info',
          message: narrationEvent.human, // 'human' field is the message
          timestamp: narrationEvent.timestamp
            ? new Date(narrationEvent.timestamp).toISOString()
            : new Date().toISOString(),
          actor: narrationEvent.actor,
          action: narrationEvent.action,
          context: narrationEvent.job_id || null, // Use job_id as context
          human: narrationEvent.human,
          fn_name: extractFnName(narrationEvent.formatted), // Extract from formatted field
          target: narrationEvent.target || null,
        }

        // Add to narration store
        useNarrationStore.getState().addEntry(keeperEvent)

        // TEAM-413: Update download progress if this is a download-related narration
        if (narrationEvent.job_id) {
          const message = narrationEvent.human
          const isDownloadRelated =
            message.includes('Download') ||
            message.includes('download') ||
            message.includes('Progress') ||
            message.includes('📥') ||
            message.includes('📊') ||
            message.includes('model_download') ||
            message.includes('worker_install')

          if (isDownloadRelated) {
            useDownloadStore.getState().updateFromNarration(narrationEvent.job_id, message)
          }
        }
      }
    },
    debug: true,
    validate: true,
  })
}
