// TEAM-292: Bee Keeper page - CLI operations interface
// Ported from web-ui.old - Only operations that cannot be done through Queen HTTP API
// TEAM-294: Connected to Tauri commands with Zustand state management
// TEAM-294: Replaced TerminalWindow with SSH targets table
// TEAM-294: Added PageContainer and dropdown menu for each row
// TEAM-294: Load real SSH targets from ~/.ssh/config via Tauri
// TEAM-294: Extracted SSH targets table into separate component

import { invoke } from "@tauri-apps/api/core";
import { PageContainer } from "@rbee/ui/molecules";
import { CommandsSidebar } from "../components/CommandsSidebar";
import { SshTargetsTable } from "../components/SshTargetsTable";
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
      
      // If there's data, format it appropriately
      if (response.data) {
        appendOutput("");
        
        // Special handling for hive-list to display as table
        if (command === "hive-list") {
          try {
            const targets = JSON.parse(response.data);
            appendOutput("┌─────────────────────┬──────────────────────┬────────────┬─────────┐");
            appendOutput("│ Host                │ Hostname:Port        │ User       │ Status  │");
            appendOutput("├─────────────────────┼──────────────────────┼────────────┼─────────┤");
            targets.forEach((target: any) => {
              const host = target.host.padEnd(19);
              const hostPort = `${target.hostname}:${target.port}`.padEnd(20);
              const user = target.user.padEnd(10);
              const status = target.status.padEnd(7);
              appendOutput(`│ ${host} │ ${hostPort} │ ${user} │ ${status} │`);
            });
            appendOutput("└─────────────────────┴──────────────────────┴────────────┴─────────┘");
          } catch (e) {
            // Fallback to raw data if parsing fails
            const dataLines = response.data.split("\n");
            dataLines.forEach((line) => {
              if (line.trim()) {
                appendOutput(line);
              }
            });
          }
        } else {
          // Default: split by lines
          const dataLines = response.data.split("\n");
          dataLines.forEach((line) => {
            if (line.trim()) {
              appendOutput(line);
            }
          });
        }
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

  // Handle refresh from SSH targets table
  const handleRefresh = async () => {
    // Trigger the hive-list command to show narration output
    await handleCommandClick("hive-list");
  };

  return (
    <div className="flex h-full w-full">
      <CommandsSidebar
        onCommandClick={handleCommandClick}
        activeCommand={activeCommand}
        disabled={isExecuting}
      />

      {/* SSH Targets Table */}
      <PageContainer
        title="SSH Targets"
        description="Available hosts from ~/.ssh/config"
        padding="lg"
      >
        <SshTargetsTable onRefresh={handleRefresh} />
      </PageContainer>
    </div>
  );
}
