// TEAM-291: Bee Keeper page - CLI operations interface
// Only operations that cannot be done through Queen HTTP API

"use client";

import { useState } from "react";
import { SidebarProvider } from "@rbee/ui/atoms";
import { TerminalWindow } from "@rbee/ui/molecules";
import { CommandsSidebar } from "@/src/components/CommandsSidebar";

export default function KeeperPage() {
  const [activeCommand, setActiveCommand] = useState<string | undefined>();
  const [output, setOutput] = useState<string>("Click a command to execute...");

  const handleCommandClick = (command: string) => {
    setActiveCommand(command);
    
    // TODO: Replace with actual command execution
    const timestamp = new Date().toLocaleTimeString();
    setOutput(`[${timestamp}] Executing: ${command}\n\nCommand execution not yet implemented.\nThis will execute the CLI command and stream output here.`);
  };

  return (
    <SidebarProvider>
      <div className="flex h-[calc(100vh-2rem)] w-full">
        <CommandsSidebar 
          onCommandClick={handleCommandClick}
          activeCommand={activeCommand}
        />

        {/* Command Output Area */}
        <div className="flex-1 p-4">
          <TerminalWindow 
            title="rbee-keeper" 
            variant="output"
            copyable
            copyText={output}
            className="h-full"
          >
            <div className="whitespace-pre-wrap">
              <span className={activeCommand ? "text-foreground" : "text-muted-foreground"}>
                {output}
              </span>
            </div>
          </TerminalWindow>
        </div>
      </div>
    </SidebarProvider>
  );
}
