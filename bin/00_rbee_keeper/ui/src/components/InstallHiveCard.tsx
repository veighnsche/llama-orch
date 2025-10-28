// TEAM-338: Card for installing hive to a new SSH target
// Uses SshHivesDataProvider with React 19 use() hook - NO useEffect needed
import { useState } from "react";
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
import { Download } from "lucide-react";
import { SshHivesDataProvider } from "../containers/SshHivesContainer";
import { useSshHivesStore } from "../store/hiveStore";
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
function InstallHiveContent() {
  const [selectedTarget, setSelectedTarget] = useState<string>("localhost");
  const { hives, install } = useSshHivesStore();
  const { isExecuting } = useCommandStore();

  const handleInstall = async () => {
    await install(selectedTarget);
  };

  // TODO: Track installed hives via store to filter them out
  const availableHives = hives;
  const isLocalhostInstalled = false; // TODO: Check from store

  return (
    <>
      <Select value={selectedTarget} onValueChange={setSelectedTarget}>
        <SelectTrigger id="install-target" className="w-full h-auto py-3">
          <SelectValue placeholder="Select target" />
        </SelectTrigger>
        <SelectContent>
          {/* Always include localhost if not installed */}
          {!isLocalhostInstalled && (
            <SelectItem value="localhost">
              <SshTargetItem name="localhost" subtitle="This machine" />
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

      {/* Install Button */}
      <Button
        onClick={handleInstall}
        disabled={isExecuting}
        className="w-full"
      >
        <Download className="mr-2 h-4 w-4" />
        Install Hive
      </Button>
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
              <InstallHiveContent />
            </SshHivesDataProvider>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
