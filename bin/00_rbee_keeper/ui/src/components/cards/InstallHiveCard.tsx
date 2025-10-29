// TEAM-338: Card for installing hive to a new SSH target
// Uses SshHivesDataProvider with React 19 use() hook - NO useEffect needed

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  DropdownMenuItem,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  SplitButton,
} from "@rbee/ui/atoms";
import { Download, FileEdit, RefreshCw } from "lucide-react";
import { useState } from "react";
import { commands } from "@/generated/bindings";
import { useCommandStore } from "../../store/commandStore";
import { useSshHives, useHiveActions, useInstalledHives } from "../../store/hiveQueries";
import type { SshHive } from "../../store/hiveQueries";

// SSH Target Select Item component
function SshTargetItem({ name, subtitle }: { name: string; subtitle: string }) {
  return (
    <div className="flex flex-col items-start">
      <span className="font-medium">{name}</span>
      <span className="text-xs text-muted-foreground">{subtitle}</span>
    </div>
  );
}

// TEAM-352: Component using new query hooks
function InstallHiveContent() {
  const [selectedTarget, setSelectedTarget] = useState<string>("");
  const { data: hives = [], refetch } = useSshHives();
  const { install } = useHiveActions();
  const { data: installedHives = [] } = useInstalledHives();
  const { isExecuting } = useCommandStore();

  const handleOpenSshConfig = async () => {
    try {
      await commands.sshOpenConfig();
    } catch (error) {
      console.error("Failed to open SSH config:", error);
    }
  };

  // TEAM-360: Filter out already installed hives (include localhost if not installed)
  const availableHives = hives.filter(
    (hive: SshHive) => !installedHives.includes(hive.host)
  );

  // Set default selection when hives load
  if (selectedTarget === "" && availableHives.length > 0) {
    setSelectedTarget(availableHives[0].host);
  }

  return (
    <>
      <Select value={selectedTarget} onValueChange={setSelectedTarget}>
        <SelectTrigger id="install-target" className="w-full h-auto py-3">
          <SelectValue placeholder="Select target" />
        </SelectTrigger>
        <SelectContent>
          {/* TEAM-360: Show all available targets including localhost if not installed */}
          {availableHives.map((hive: SshHive) => (
            <SelectItem key={hive.host} value={hive.host}>
              <SshTargetItem
                name={hive.host}
                subtitle={`${hive.user}@${hive.hostname}:${hive.port}`}
              />
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Install Button with Actions */}
      <SplitButton
        onClick={() => install(selectedTarget)}
        icon={<Download className="h-4 w-4" />}
        disabled={isExecuting}
        className="w-full"
        dropdownContent={
          <>
            <DropdownMenuItem onClick={() => refetch()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </DropdownMenuItem>
            <DropdownMenuItem onClick={handleOpenSshConfig}>
              <FileEdit className="mr-2 h-4 w-4" />
              Edit SSH Config
            </DropdownMenuItem>
          </>
        }
      >
        Install Hive
      </SplitButton>
    </>
  );
}

export function InstallHiveCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Install Hive</CardTitle>
        <CardDescription>
          Choose a target to install the Hive worker manager
        </CardDescription>
      </CardHeader>
      <div className="flex-1" />
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
            <InstallHiveContent />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
