// TEAM-336: Narration panel - displays real-time narration events from Rust backend
// Listens to "narration" events emitted by custom tracing layer

import { useEffect, useState, useRef } from "react";
import { listen } from "@tauri-apps/api/event";
import type { NarrationEvent } from "../generated/bindings";
import { ScrollArea } from "@rbee/ui/atoms";

interface NarrationEntry extends NarrationEvent {
  id: number;
}

export function NarrationPanel() {
  const [entries, setEntries] = useState<NarrationEntry[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const idCounter = useRef(0);

  // Auto-scroll to bottom when new entries arrive
  useEffect(() => {
    if (scrollRef.current) {
      const scrollElement = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [entries]);

  // Listen to narration events from Rust backend
  // TEAM-337: Requires core:event:allow-listen permission in tauri.conf.json
  useEffect(() => {
    const unlisten = listen<NarrationEvent>("narration", (event) => {
      setEntries((prev) => [
        ...prev,
        {
          ...event.payload,
          id: idCounter.current++,
        },
      ]);
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, []);

  // Clear all entries
  const handleClear = () => {
    setEntries([]);
    idCounter.current = 0;
  };

  // Format timestamp to HH:MM:SS
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  // Get color for log level (unused but kept for future use)
  // const getLevelColor = (level: string) => {
  //   switch (level.toLowerCase()) {
  //     case "error":
  //       return "text-red-500";
  //     case "warn":
  //       return "text-yellow-500";
  //     case "info":
  //       return "text-blue-500";
  //     case "debug":
  //       return "text-gray-500";
  //     default:
  //       return "text-foreground";
  //   }
  // };

  // Get level badge style
  const getLevelBadge = (level: string) => {
    const baseClasses = "px-1.5 py-0.5 rounded text-xs font-mono font-semibold";
    switch (level.toLowerCase()) {
      case "error":
        return `${baseClasses} bg-red-500/10 text-red-500`;
      case "warn":
        return `${baseClasses} bg-yellow-500/10 text-yellow-500`;
      case "info":
        return `${baseClasses} bg-blue-500/10 text-blue-500`;
      case "debug":
        return `${baseClasses} bg-gray-500/10 text-gray-500`;
      default:
        return `${baseClasses} bg-muted text-muted-foreground`;
    }
  };

  // TEAM-336: Test button to verify narration pipeline
  const handleTest = async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("test_narration");
    } catch (error) {
      console.error("[NarrationPanel] Test failed:", error);
    }
  };

  return (
    <div className="w-80 border-l border-border bg-background flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <h2 className="text-sm font-semibold">Narration</h2>
        <div className="flex gap-2">
          <button
            onClick={handleTest}
            className="text-xs text-blue-500 hover:text-blue-600 transition-colors"
            title="Test narration events"
          >
            Test
          </button>
          <button
            onClick={handleClear}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            title="Clear all entries"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Entries list */}
      <ScrollArea className="flex-1" ref={scrollRef}>
        <div className="p-2 space-y-2">
          {entries.length === 0 ? (
            <div className="text-center text-sm text-muted-foreground py-8">
              Waiting for events...
            </div>
          ) : (
            entries.map((entry) => (
              <div
                key={entry.id}
                className="p-2 rounded-md bg-muted/30 hover:bg-muted/50 transition-colors text-xs space-y-1"
              >
                {/* Timestamp and level */}
                <div className="flex items-center justify-between gap-2">
                  <span className="text-muted-foreground font-mono">
                    {formatTime(entry.timestamp)}
                  </span>
                  <span className={getLevelBadge(entry.level)}>
                    {entry.level.toUpperCase()}
                  </span>
                </div>

                {/* Message */}
                <div className="text-foreground break-words font-mono leading-relaxed">
                  {entry.message}
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      {/* Footer stats */}
      <div className="px-4 py-2 border-t border-border text-xs text-muted-foreground">
        {entries.length} {entries.length === 1 ? "entry" : "entries"}
      </div>
    </div>
  );
}
