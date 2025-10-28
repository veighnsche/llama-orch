// TEAM-338: Card for installing hive to a new SSH target
// DEPRECATED: Use HiveInstallCard.tsx instead (uses Zustand store)
import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Button,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@rbee/ui/atoms";
import { Download, Loader2, AlertCircle } from "lucide-react";
import { useSshHivesStore, type SshHive } from "../store/sshHivesStore";
import { useInstallationStore } from "../store/hiveStore";

interface InstallHiveCardProps {
  onInstall: (targetId: string) => void;
  disabled?: boolean;
}

// SSH Target Select Item component
function SshTargetItem({ name, subtitle }: { name: string; subtitle: string }) {
  return (
    <div className="flex flex-col items-start">
      <span className="font-medium">{name}</span>
      <span className="text-xs text-muted-foreground">{subtitle}</span>
    </div>
  );
}

export function InstallHiveCard({
  onInstall,
  disabled = false,
}: InstallHiveCardProps) {
  const [selectedTarget, setSelectedTarget] = useState<string>("localhost");
  const { installedHives } = useInstallationStore();
  const { hives, isLoading, error, fetchHives } = useSshHivesStore();

  // Fetch hives on mount
  useEffect(() => {
    fetchHives();
  }, [fetchHives]);

  const handleInstall = () => {
    onInstall(selectedTarget);
  };

  // Filter out already installed hives
  const availableHives = hives.filter(
    (hive) => !installedHives.includes(hive.host),
  );

  // Check if localhost is already installed
  const isLocalhostInstalled = installedHives.includes("localhost");

  return (
    <Card>
      <CardHeader>
        <CardTitle>Install Hive</CardTitle>
        <CardDescription>
          Choose a target to install the Hive worker manager
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* SSH Target Selection */}
          <div className="space-y-2">
            <label
              htmlFor="install-target"
              className="text-sm font-medium text-foreground"
            >
              Target
            </label>

            {/* Loading state */}
            {isLoading && hives.length === 0 ? (
              <Select disabled>
                <SelectTrigger id="install-target" className="w-full">
                  <SelectValue placeholder="Loading targets..." />
                </SelectTrigger>
              </Select>
            ) : error ? (
              /* Error state */
              <div className="flex items-center gap-2 text-destructive text-sm p-3 border border-destructive/50 rounded-md">
                <AlertCircle className="h-4 w-4" />
                <p>{error}</p>
              </div>
            ) : (
              /* Success state */
              <Select
                value={selectedTarget}
                onValueChange={setSelectedTarget}
              >
                <SelectTrigger
                  id="install-target"
                  className="w-full h-auto py-3"
                >
                  <SelectValue placeholder="Select target" />
                </SelectTrigger>
                <SelectContent>
                  {/* Always include localhost if not installed */}
                  {!isLocalhostInstalled && (
                    <SelectItem value="localhost">
                      <SshTargetItem
                        name="localhost"
                        subtitle="This machine"
                      />
                    </SelectItem>
                  )}
                  {/* Dynamic SSH targets (filtered) */}
                  {availableHives.map((hive) => (
                    <SelectItem key={hive.host} value={hive.host}>
                      <SshTargetItem
                        name={hive.host}
                        subtitle={`${hive.user}@${hive.hostname}:${hive.port}`}
                      />
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
          </div>

          {/* Install Button */}
          <Button
            onClick={handleInstall}
            disabled={disabled || isLoading || !!error}
            className="w-full"
          >
            <Download className="mr-2 h-4 w-4" />
            Install Hive
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
