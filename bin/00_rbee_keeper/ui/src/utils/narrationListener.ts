// TEAM-XXX: Listen for narration events from Queen iframe
// Receives postMessage events and adds them to narration store
// TEAM-350: Map Queen's narration format to Keeper's format

import { useNarrationStore } from "../store/narrationStore";
import type { NarrationEvent } from "../generated/bindings";

// TEAM-350: Queen's narration event format (from backend SSE)
export interface QueenNarrationEvent {
  actor: string;
  action: string;
  human: string; // The message text
  level?: string;
  timestamp?: number;
  job_id?: string;
  target?: string;
  formatted?: string;
}

export interface NarrationMessage {
  type: "QUEEN_NARRATION";
  payload: QueenNarrationEvent;
  source: "queen-rbee";
  timestamp: number;
}

/**
 * Setup listener for narration events from Queen iframe
 * Call this once at app startup
 */
export function setupNarrationListener(): () => void {
  const handleMessage = (event: MessageEvent) => {
    // TEAM-350: Security - Verify origin is Queen
    // Dev: Queen runs on :7834 (Vite dev server)
    // Prod: Queen runs on :7833 (embedded in backend)
    const allowedOrigins = [
      "http://localhost:7833", // Prod: Queen backend
      "http://localhost:7834", // Dev: Queen Vite dev server
    ];

    if (!allowedOrigins.includes(event.origin)) {
      console.warn(
        "[Keeper] Rejected message from unknown origin:",
        event.origin,
      );
      return;
    }

    // Filter for Queen narration events
    if (event.data?.type === "QUEEN_NARRATION") {
      const message = event.data as NarrationMessage;
      const queenEvent = message.payload;

      console.log("[Keeper] Received narration from Queen:", queenEvent);

      // TEAM-350: Extract function name from formatted field (contains ANSI codes)
      // Format: "[1mfunction_name[0m [2maction[0m\nmessage"
      // The [1m...[0m is the function name in bold
      const extractFnName = (formatted?: string): string | null => {
        if (!formatted) return null;
        
        // Remove the literal escape character representations and use actual escape codes
        // The formatted string contains actual ANSI escape sequences, not \u001b literals
        // Match text between ESC[1m (bold) and ESC[0m (reset)
        const match = formatted.match(/\x1b\[1m([^\x1b]+)\x1b\[0m/);
        return match ? match[1] : null;
      };

      // TEAM-350: Map Queen's format to Keeper's format
      const keeperEvent: NarrationEvent = {
        level: queenEvent.level || "info",
        message: queenEvent.human, // Queen's 'human' field is the message
        timestamp: queenEvent.timestamp
          ? new Date(queenEvent.timestamp).toISOString()
          : new Date().toISOString(),
        actor: queenEvent.actor,
        action: queenEvent.action,
        context: queenEvent.job_id || null, // Use job_id as context
        human: queenEvent.human,
        fn_name: extractFnName(queenEvent.formatted), // Extract from formatted field
        target: queenEvent.target || null,
      };

      // Add to narration store
      useNarrationStore.getState().addEntry(keeperEvent);
    }
  };

  // Add listener
  window.addEventListener("message", handleMessage);

  // Return cleanup function
  return () => {
    window.removeEventListener("message", handleMessage);
  };
}
