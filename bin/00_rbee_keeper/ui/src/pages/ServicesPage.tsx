// TEAM-294: Keeper page - Status cards and SSH targets table
// TEAM-295: Added action buttons to Queen and Hive cards (icon-only with tooltips)
// TEAM-296: Wired up queen lifecycle operations (install, update, uninstall)
//
// Queen Lifecycle Operations:
// - start: Run the queen daemon
// - stop: Stop the queen daemon
// - install: Build from git repo (cargo build --release) and install to ~/.local/bin (errors if already installed)
// - update: Rebuild from source (cargo build --release)
// - uninstall: Remove binary from ~/.local/bin (errors if not installed)

import { invoke } from "@tauri-apps/api/core";
import { PageContainer } from "@rbee/ui/molecules";
import { SshHivesContainer } from "../components/SshHivesContainer";
import { ServiceCard } from "../components/ServiceCard";
import { useCommandStore } from "../store/commandStore";

export default function KeeperPage() {
  const { setActiveCommand, isExecuting, setIsExecuting } = useCommandStore();

  const handleCommand = async (command: string) => {
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
        case "queen-install":
          // TEAM-296: Build from source and install to ~/.local/bin
          await invoke("queen_install", { binary: null });
          break;
        case "queen-update":
          // TEAM-296: Rebuild from source (same as rebuild)
          await invoke("queen_rebuild", { withLocalHive: false });
          break;
        case "queen-uninstall":
          // TEAM-296: Remove binary from ~/.local/bin
          await invoke("queen_uninstall");
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
        case "hive-install":
          // TODO: Implement hive install
          console.log("Hive install not yet implemented");
          break;
        case "hive-update":
          // TODO: Implement hive update
          console.log("Hive update not yet implemented");
          break;
        case "hive-uninstall":
          // TODO: Implement hive uninstall
          console.log("Hive uninstall not yet implemented");
          break;
      }
    } catch (error) {
      console.error("Command failed:", error);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <PageContainer
      title="Services"
      description="Manage Queen, Hive, and SSH connections"
      padding="lg"
      helperText={[
        {
          title: "Queen",
          description:
            "routes inference jobs to the right worker in the right hive. Start Queen first to enable job routing.",
        },
        {
          title: "Hive",
          description:
            "manages worker lifecycle and catalogs (models from HuggingFace, worker binaries). Start localhost hive to see local models and workers. Use SSH targets above to start remote hives and access their catalogs.",
        },
      ]}
    >
      <div className="space-y-6">
        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ServiceCard
            title="Queen"
            description="Smart API server"
            details="Job router that dispatches inference requests to workers in the correct hive"
            servicePrefix="queen"
            status="unknown" // TODO: Implement state detection
            onCommandClick={handleCommand}
            disabled={isExecuting}
          />

          <ServiceCard
            title="Hive"
            description="Local Worker Manager"
            details="Manages workers and catalogs (models, worker binaries) on this machine"
            servicePrefix="hive"
            status="unknown" // TODO: Implement state detection
            onCommandClick={handleCommand}
            disabled={isExecuting}
          />
        </div>

        {/* SSH Hives Table */}
        <div>
          <h3 className="text-lg font-semibold mb-4">SSH Hives</h3>
          <SshHivesContainer />
        </div>
      </div>
    </PageContainer>
  );
}
