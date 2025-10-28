// TEAM-294: Keeper page - Status cards and SSH targets table
// TEAM-295: Added action buttons to Queen and Hive cards (icon-only with tooltips)
// TEAM-296: Wired up queen lifecycle operations (install, update, uninstall)
// Updated: Replaced ServiceCard with dedicated QueenCard and HiveCard components
//
// Queen Lifecycle Operations:
// - start: Run the queen daemon
// - stop: Stop the queen daemon
// - install: Build from git repo (cargo build --release) and install to ~/.local/bin (errors if already installed)
// - update: Rebuild from source (cargo build --release)
// - uninstall: Remove binary from ~/.local/bin (errors if not installed)

import { invoke } from "@tauri-apps/api/core";
import { PageContainer } from "@rbee/ui/molecules";
import { QueenCard } from "../components/QueenCard";
import { InstalledHiveCard } from "../components/InstalledHiveCard";
import { InstallHiveCard } from "../components/InstallHiveCard";
import { SshHivesDataProvider, type SshHive } from "../components/SshHivesContainer";
import { useCommandStore } from "../store/commandStore";
import { useInstallationStore } from "../store/hiveStore";

export default function KeeperPage() {
  const { setActiveCommand, isExecuting, setIsExecuting } = useCommandStore();
  const { installedHives, addInstalledHive, removeInstalledHive } =
    useInstallationStore();

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

  const handleHiveCommand = async (command: string, targetId: string) => {
    setActiveCommand(command);
    setIsExecuting(true);

    try {
      switch (command) {
        case "hive-start":
          await invoke("hive_start", {
            host: targetId,
            installDir: null,
            port: 7835,
          });
          break;
        case "hive-stop":
          await invoke("hive_stop", { host: targetId });
          break;
        case "hive-install":
          // TODO: Implement hive install
          console.log(`Hive install not yet implemented for ${targetId}`);
          break;
        case "hive-update":
          // TODO: Implement hive update
          console.log(`Hive update not yet implemented for ${targetId}`);
          break;
        case "hive-uninstall":
          // Remove from installed hives when uninstalled
          removeInstalledHive(targetId);
          // TODO: Implement actual uninstall
          console.log(`Hive uninstall not yet implemented for ${targetId}`);
          break;
      }
    } catch (error) {
      console.error("Command failed:", error);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleInstallHive = async (targetId: string) => {
    setActiveCommand("hive-install");
    setIsExecuting(true);

    try {
      // TODO: Implement actual install
      await invoke("hive_start", {
        host: targetId,
        installDir: null,
        port: 7835,
      });
      // Add to installed hives after successful install
      addInstalledHive(targetId);
    } catch (error) {
      console.error("Install failed:", error);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <PageContainer
      title="Services"
      description="Manage Queen and Hive services"
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
            "manages worker lifecycle and catalogs (models from HuggingFace, worker binaries). Start localhost hive to see local models and workers. Use SSH targets to start remote hives and access their catalogs.",
        },
      ]}
    >
      <div className="space-y-4 sm:space-y-6">
        {/* Queen Card */}
        <QueenCard
          status="unknown" // TODO: Implement state detection
          onCommandClick={handleCommand}
          disabled={isExecuting}
        />

        {/* Installed Hives - Need SSH data to show names */}
        <SshHivesDataProvider fallback={null}>
          {(hives) => (
            <>
              {installedHives.map((targetId) => {
                // Find the hive data for this target
                const hiveData = hives.find((h) => h.host === targetId);
                const isLocalhost = targetId === "localhost";

                const targetName = isLocalhost ? "Hive (localhost)" : hiveData?.host || targetId;
                const targetSubtitle = isLocalhost
                  ? "Local Worker Manager"
                  : hiveData
                  ? `${hiveData.user}@${hiveData.hostname}:${hiveData.port}`
                  : "Worker Manager";

                return (
                  <InstalledHiveCard
                    key={targetId}
                    targetId={targetId}
                    targetName={targetName}
                    targetSubtitle={targetSubtitle}
                    status="unknown" // TODO: Implement state detection
                    onCommandClick={handleHiveCommand}
                    disabled={isExecuting}
                  />
                );
              })}
            </>
          )}
        </SshHivesDataProvider>

        {/* Install New Hive Card */}
        <InstallHiveCard
          onInstall={handleInstallHive}
          disabled={isExecuting}
        />
      </div>
    </PageContainer>
  );
}
