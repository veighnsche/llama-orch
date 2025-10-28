// TEAM-334: Shell component - contains titlebar and sidebar
// Provides the main layout structure for the app
// Pages are rendered inside this shell
// TEAM-336: Added NarrationPanel on the right side
// TEAM-339: Made panels resizable with react-resizable-panels library

import { type ReactNode } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import { CustomTitlebar } from "./CustomTitlebar";
import { KeeperSidebar } from "./KeeperSidebar";
import { NarrationPanel } from "./NarrationPanel";

interface ShellProps {
  children: ReactNode;
}

export function Shell({ children }: ShellProps) {
  return (
    <div className="fixed inset-0 flex flex-col bg-background text-foreground">
      {/* Titlebar - fixed height */}
      <CustomTitlebar />

      {/* Main content area - left sidebar + page content + right narration panel */}
      <PanelGroup direction="horizontal" autoSaveId="keeper-layout" className="flex-1">
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
          <main className="h-full overflow-y-auto">{children}</main>
        </Panel>

        <PanelResizeHandle className="w-1 bg-transparent hover:bg-blue-500 transition-colors" />

        {/* Right panel - narration stream - resizable */}
        <Panel
          id="narration"
          defaultSize={25}
          minSize={20}
          maxSize={40}
          order={3}
        >
          <NarrationPanel />
        </Panel>
      </PanelGroup>
    </div>
  );
}
