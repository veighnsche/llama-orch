// TEAM-334: Shell component - contains titlebar and sidebar
// Provides the main layout structure for the app
// Pages are rendered inside this shell

import { type ReactNode } from "react";
import { CustomTitlebar } from "./CustomTitlebar";
import { KeeperSidebar } from "./KeeperSidebar";

interface ShellProps {
  children: ReactNode;
}

export function Shell({ children }: ShellProps) {
  return (
    <div className="fixed inset-0 flex flex-col bg-background text-foreground">
      {/* Titlebar - fixed height */}
      <CustomTitlebar />

      {/* Main content area - sidebar + page content */}
      <div className="flex-1 flex overflow-hidden">
        <KeeperSidebar />

        {/* Page content area - scrollable container */}
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
