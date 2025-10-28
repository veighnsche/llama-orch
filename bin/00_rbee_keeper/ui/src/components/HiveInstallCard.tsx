// TEAM-338: Card for installing hive to SSH targets
// Uses SshHivesDataProvider with React 19 use() hook - NO useEffect needed
// Gets data from useSshHivesStore() - NO props drilling
// SplitButton with Install Hive (main), Refresh, Open SSH Config (dropdown)

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  SplitButton,
  DropdownMenuItem,
  DropdownMenuSeparator,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@rbee/ui/atoms";
import { Download, RefreshCw, FileText } from "lucide-react";
import { SshHivesDataProvider } from "../containers/SshHivesContainer";
import { useSshHivesStore, type SshHive } from "../store/sshHivesStore";
import { useCommandStore } from "../store/commandStore";

// SSH Target Select Item component
function SshTargetItem({ name, subtitle }: { name: string; subtitle: string }) {
  return (
    <div className="flex flex-col items-start">
      <span className="font-medium">{name}</span>
      <span className="text-xs text-muted-foreground">{subtitle}</span>
    </div>
  );
}

// Inner component that reads from store
function HiveInstallContent() {
  const [selectedTarget, setSelectedTarget] = useState<string>("localhost");
  const { hives, refresh } = useSshHivesStore();
  const { isExecuting } = useCommandStore();

  const handleInstall = () => {
    // TODO: Wire up hive install command
    console.log("Install hive on:", selectedTarget);
  };

  const handleOpenSshConfig = () => {
    // TODO: Wire up open SSH config command
    console.log("Open SSH config");
  };

  return (
    <>
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
          {/* Always include localhost */}
          <SelectItem value="localhost">
            <SshTargetItem
              name="localhost"
              subtitle="This machine"
            />
          </SelectItem>
          {/* Dynamic SSH targets */}
          {hives.map((hive) => (
            <SelectItem key={hive.host} value={hive.host}>
              <SshTargetItem
                name={hive.host}
                subtitle={`${hive.user}@${hive.hostname}:${hive.port}`}
              />
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Install Button with Dropdown */}
      <SplitButton
        variant="default"
        size="default"
        icon={<Download className="h-4 w-4" />}
        onClick={handleInstall}
        disabled={isExecuting}
        className="w-full"
        dropdownContent={
          <>
            <DropdownMenuItem onClick={refresh}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleOpenSshConfig}>
              <FileText className="mr-2 h-4 w-4" />
              Open SSH Config
            </DropdownMenuItem>
          </>
        }
      >
        Install Hive
      </SplitButton>
    </>
  );
}

export function HiveInstallCard() {
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

            <SshHivesDataProvider
              fallback={
                <Select disabled>
                  <SelectTrigger id="install-target" className="w-full">
                    <SelectValue placeholder="Loading targets..." />
                  </SelectTrigger>
                </Select>
              }
            >
              <HiveInstallContent />
            </SshHivesDataProvider>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
