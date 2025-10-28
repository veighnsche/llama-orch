// TEAM-334: Shell component - contains titlebar and sidebar
// Provides the main layout structure for the app
// Pages are rendered inside this shell
// TEAM-336: Added NarrationPanel on the right side

import { type ReactNode } from "react";
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
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar - navigation */}
        <KeeperSidebar />

        {/* Page content area - scrollable container */}
        <main className="flex-1 overflow-y-auto">{children}</main>

        {/* Right panel - narration stream */}
        <NarrationPanel />
      </div>
    </div>
  );
}
