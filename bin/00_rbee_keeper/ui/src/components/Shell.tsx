// TEAM-334: Shell component - contains titlebar and sidebar
// Provides the main layout structure for the app
// Pages are rendered inside this shell
// TEAM-336: Added NarrationPanel on the right side
// TEAM-339: Made panels resizable with react-resizable-panels library

import { listen } from "@tauri-apps/api/event";
import { MessageSquare } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import type { NarrationEvent } from "../generated/bindings";
import { useNarrationStore } from "../store/narrationStore";
import { CustomTitlebar } from "./CustomTitlebar";
import { KeeperSidebar } from "./KeeperSidebar";
import { NarrationPanel } from "./NarrationPanel";

interface ShellProps {
  children: ReactNode;
}

export function Shell({ children }: ShellProps) {
  const { showNarration, setShowNarration, addEntry } = useNarrationStore();

  // TEAM-339: Listen to narration events at Shell level (always active)
  // This ensures we don't miss events when panel is closed
  useEffect(() => {
    const unlisten = listen<NarrationEvent>("narration", (event) => {
      addEntry(event.payload);
    });

    return () => {
      unlisten.then((fn) => fn());
    };
  }, [addEntry]);

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-background font-sans">
      {/* Titlebar - fixed height */}
      <CustomTitlebar />

      {/* Main content area - left sidebar + page content + right narration panel */}
      <PanelGroup
        direction="horizontal"
        autoSaveId="keeper-layout"
        className="flex-1"
      >
        {/* Left sidebar - navigation - resizable */}
        <Panel
          id="sidebar"
          defaultSize={20}
          minSize={15}
          maxSize={30}
          order={1}
        >
          <KeeperSidebar />
        </Panel>

        <PanelResizeHandle className="w-1 bg-transparent hover:bg-blue-500 transition-colors" />

        {/* Page content area - scrollable container - takes remaining space */}
        <Panel id="main" minSize={30} order={2}>
          <div className="relative h-full">
            <main className="h-full overflow-y-auto">{children}</main>

            {/* TEAM-339: Show open narration button when panel is closed */}
            {!showNarration && (
              <button
                onClick={() => setShowNarration(true)}
                className="absolute top-4 right-4 p-2 rounded-lg text-primary-foreground hover:bg-primary/90 shadow-lg transition-colors"
                aria-label="Open narration panel"
              >
                <MessageSquare className="h-5 w-5" />
              </button>
            )}
          </div>
        </Panel>

        {showNarration && (
          <>
            <PanelResizeHandle className="w-1 bg-transparent hover:bg-blue-500 transition-colors" />

            {/* Right panel - narration stream - resizable */}
            <Panel
              id="narration"
              defaultSize={20}
              minSize={20}
              maxSize={40}
              order={3}
            >
              <NarrationPanel onClose={() => setShowNarration(false)} />
            </Panel>
          </>
        )}
      </PanelGroup>
    </div>
  );
}
