// TEAM-294: Keeper page - Status cards and SSH targets table
// TEAM-295: Added action buttons to Queen and Hive cards (icon-only with tooltips)

import { PageContainer } from "@rbee/ui/molecules";
import { Card, CardContent, CardHeader, CardTitle } from "@rbee/ui/atoms";
import { SshTargetsTable } from "../components/SshTargetsTable";
import { ServiceActionButtons } from "../components/ServiceActionButtons";
import { useCommandStore } from "../store/commandStore";

export default function KeeperPage() {
  const { setActiveCommand, isExecuting } = useCommandStore();

  const handleCommand = (command: string) => {
    setActiveCommand(command);
  };

  return (
    <PageContainer
      title="Dashboard"
      description="System status and SSH targets"
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
