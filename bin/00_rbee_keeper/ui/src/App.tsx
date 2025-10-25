// TEAM-294: rbee-keeper GUI - Main application component
import { invoke } from "@tauri-apps/api/core";
import KeeperPage from "./pages/KeeperPage";
import { KeeperSidebar } from "./components/Sidebar";
import { SidebarProvider } from "@rbee/ui/atoms";
import { useCommandStore } from "./store/commandStore";

function App() {
  const { activeCommand, isExecuting, setActiveCommand, setIsExecuting } =
    useCommandStore();

  const handleCommandClick = async (command: string) => {
    setActiveCommand(command);
    setIsExecuting(true);

    try {
      switch (command) {
        case "queen-start":
          await invoke("queen_start");
          break;
        case "queen-stop":
          await invoke("queen_stop");
          break;
        case "queen-status":
          await invoke("queen_status");
          break;
        case "queen-info":
          await invoke("queen_info");
          break;
        case "queen-rebuild":
          await invoke("queen_rebuild", { withLocalHive: false });
          break;
        case "hive-start":
          await invoke("hive_start", {
            host: "localhost",
            installDir: null,
            port: 7835,
          });
          break;
        case "hive-stop":
          await invoke("hive_stop", { host: "localhost" });
          break;
        case "hive-status":
          await invoke("hive_status", { alias: "localhost" });
          break;
        case "hive-list":
          await invoke("hive_list");
          break;
      }
    } catch (error) {
      console.error("Command failed:", error);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <SidebarProvider>
      <div className="h-screen w-screen overflow-hidden bg-background text-foreground flex">
        <KeeperSidebar
          onCommandClick={handleCommandClick}
          activeCommand={activeCommand}
          disabled={isExecuting}
        />
        <KeeperPage />
      </div>
    </SidebarProvider>
  );
}

export default App;
