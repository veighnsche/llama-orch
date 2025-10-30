// TEAM-352: FIXED - Listen for narration events from service iframes
// Uses @rbee/iframe-bridge for proper message validation
// Uses @rbee/narration-client types for type safety
// Uses @rbee/shared-config for port configuration

import { createMessageReceiver } from "@rbee/iframe-bridge";
import type { BackendNarrationEvent } from "@rbee/narration-client";
import { getAllowedOrigins } from "@rbee/shared-config";
import { useNarrationStore } from "../store/narrationStore";
import type { NarrationEvent } from "../generated/bindings";

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
      if (message.type === "NARRATION_EVENT") {
        const narrationEvent = message.payload as BackendNarrationEvent;

        console.log("[Keeper] Received narration:", narrationEvent);

        // TEAM-352: Extract function name from formatted field (contains ANSI codes)
        // Format: "\x1b[1mfunction_name\x1b[0m \x1b[2maction\x1b[0m\nmessage"
        // The \x1b[1m...\x1b[0m is the function name in bold
        const extractFnName = (formatted?: string): string | null => {
          if (!formatted) return null;
          
          // Match text between ESC[1m (bold) and ESC[0m (reset)
          const match = formatted.match(/\x1b\[1m([^\x1b]+)\x1b\[0m/);
          return match ? match[1] : null;
        };

        // TEAM-352: Map shared narration format to Keeper's format
        const keeperEvent: NarrationEvent = {
          level: narrationEvent.level || "info",
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
        };

        // Add to narration store
        useNarrationStore.getState().addEntry(keeperEvent);
      }
    },
    debug: true,
    validate: true,
  });
}
