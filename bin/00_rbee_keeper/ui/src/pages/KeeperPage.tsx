// TEAM-292: Bee Keeper page - CLI operations interface
// Ported from web-ui.old - Only operations that cannot be done through Queen HTTP API
// TEAM-294: Connected to Tauri commands with Zustand state management

import { invoke } from "@tauri-apps/api/core";
import { TerminalWindow } from "@rbee/ui/molecules";
import { CommandsSidebar } from "../components/CommandsSidebar";
import { useCommandStore } from "../store/commandStore";

interface CommandResponse {
  success: boolean;
  message: string;
  data?: string;
}

export default function KeeperPage() {
  const {
    activeCommand,
    isExecuting,
    outputLines,
    setActiveCommand,
    setIsExecuting,
    appendOutput,
    clearOutput,
  } = useCommandStore();

  const handleCommandClick = async (command: string) => {
    // Clear previous output and set new command
    clearOutput();
    setActiveCommand(command);
    setIsExecuting(true);

    const timestamp = new Date().toLocaleTimeString();
    appendOutput(`[${timestamp}] Executing: ${command}...`);
    appendOutput("");

    try {
      let result: string;

      // Map command IDs to Tauri commands
      switch (command) {
        // Queen commands
        case "queen-start":
          result = await invoke<string>("queen_start");
          break;
        case "queen-stop":
          result = await invoke<string>("queen_stop");
          break;
        case "queen-status":
          result = await invoke<string>("queen_status");
          break;
        case "queen-info":
          result = await invoke<string>("queen_info");
          break;
        case "queen-rebuild":
          result = await invoke<string>("queen_rebuild", {
            withLocalHive: false,
          });
          break;

        // Hive commands (using localhost as default)
        case "hive-start":
          result = await invoke<string>("hive_start", {
            host: "localhost",
            installDir: null,
            port: 7835,
          });
          break;
        case "hive-stop":
          result = await invoke<string>("hive_stop", { host: "localhost" });
          break;
        case "hive-status":
          result = await invoke<string>("hive_status", { alias: "localhost" });
          break;
        case "hive-list":
          result = await invoke<string>("hive_list");
          break;

        default:
          result = JSON.stringify({
            success: false,
            message: `Unknown command: ${command}`,
          });
      }

      // Parse the response
      const response: CommandResponse = JSON.parse(result);
      const statusIcon = response.success ? "✅" : "❌";
      
      appendOutput("");
      appendOutput(`[${timestamp}] ${statusIcon} ${response.message}`);
      
      // If there's data, split it into lines and append each
      if (response.data) {
        appendOutput("");
        const dataLines = response.data.split("\n");
        dataLines.forEach((line) => {
          if (line.trim()) {
            appendOutput(line);
          }
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      appendOutput("");
      appendOutput(`[${timestamp}] ❌ Error: ${errorMessage}`);
    } finally {
      setIsExecuting(false);
    }
  };

  // Join output lines for copy functionality
  const outputText = outputLines.join("\n");

  return (
    <div className="flex h-full w-full">
      <CommandsSidebar
        onCommandClick={handleCommandClick}
        activeCommand={activeCommand}
        disabled={isExecuting}
      />

      {/* Command Output Area */}
      <div className="flex-1 p-4">
        <TerminalWindow
          title="rbee-keeper"
          variant="output"
          copyable
          copyText={outputText}
          className="h-full"
        >
          <div className="font-mono text-sm">
            {outputLines.map((line, index) => (
              <div
                key={index}
                className={
                  activeCommand ? "text-foreground" : "text-muted-foreground"
                }
              >
                {line || "\u00A0"} {/* Non-breaking space for empty lines */}
              </div>
            ))}
          </div>
        </TerminalWindow>
      </div>
    </div>
  );
}
