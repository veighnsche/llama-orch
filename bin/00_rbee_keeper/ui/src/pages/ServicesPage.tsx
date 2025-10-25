// TEAM-294: Keeper page - Status cards and SSH targets table
// TEAM-295: Added action buttons to Queen and Hive cards (icon-only with tooltips)

import { invoke } from "@tauri-apps/api/core";
import { PageContainer } from "@rbee/ui/molecules";
import { Card, CardContent, CardHeader, CardTitle } from "@rbee/ui/atoms";
import { SshTargetsTable } from "../components/SshTargetsTable";
import { ServiceActionButtons } from "../components/ServiceActionButtons";
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
          // TODO: Implement queen install
          console.log("Queen install not yet implemented");
          break;
        case "queen-update":
          // TODO: Implement queen update
          console.log("Queen update not yet implemented");
          break;
        case "queen-uninstall":
          // TODO: Implement queen uninstall
          console.log("Queen uninstall not yet implemented");
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
    >
      <div className="space-y-6">
        {/* Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Queen</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Manage the Queen orchestrator service
                </p>
                <ServiceActionButtons
                  servicePrefix="queen"
                  onCommandClick={handleCommand}
                  disabled={isExecuting}
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Hive (localhost)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Manage the local Hive service
                </p>
                <ServiceActionButtons
                  servicePrefix="hive"
                  onCommandClick={handleCommand}
                  disabled={isExecuting}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* SSH Targets Table */}
        <div>
          <h3 className="text-lg font-semibold mb-4">SSH Hives</h3>
          <SshTargetsTable />
        </div>
      </div>
    </PageContainer>
  );
}
