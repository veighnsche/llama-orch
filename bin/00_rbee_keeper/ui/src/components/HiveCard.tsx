// Hive service card with lifecycle controls and SSH target selection
import { useState } from "react";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@rbee/ui/atoms";
import { ServiceActionButtons } from "./ServiceActionButtons";
import { useHiveStore } from "../store/hiveStore";
import { SshHivesDataProvider, type SshHive } from "./SshHivesContainer";

export type ServiceStatus =
  | "healthy"
  | "unhealthy"
  | "stopped"
  | "not-installed"
  | "out-of-date"
  | "unknown"
  | "checking";

interface HiveCardProps {
  status?: ServiceStatus;
  onCommandClick: (command: string) => void;
  onStatusChange?: (status: ServiceStatus) => void;
  disabled?: boolean;
}

const STATUS_CONFIG: Record<
  ServiceStatus,
  {
    label: string;
    variant: "default" | "destructive" | "secondary" | "outline";
  }
> = {
  healthy: { label: "Healthy", variant: "default" },
  unhealthy: { label: "Unhealthy", variant: "destructive" },
  stopped: { label: "Stopped", variant: "secondary" },
  "not-installed": { label: "Not Installed", variant: "secondary" },
  "out-of-date": { label: "Update Available", variant: "outline" },
  unknown: { label: "Unknown", variant: "secondary" },
  checking: { label: "Checking...", variant: "outline" },
};

const HEALTH_URL = "http://localhost:7835/health";

// SSH Target Select Item component
function SshTargetItem({ name, subtitle }: { name: string; subtitle: string }) {
  return (
    <div className="flex flex-col items-start">
      <span className="font-medium">{name}</span>
      <span className="text-xs text-muted-foreground">{subtitle}</span>
    </div>
  );
}

export function HiveCard({
  status = "unknown",
  onCommandClick,
  onStatusChange,
  disabled = false,
}: HiveCardProps) {
  const [localStatus, setLocalStatus] = useState<ServiceStatus>(status);
  const { selectedTarget, setSelectedTarget } = useHiveStore();
  const currentStatus = status !== "unknown" ? status : localStatus;
  const statusConfig = STATUS_CONFIG[currentStatus];

  const handleHealthCheck = async () => {
    if (currentStatus === "checking") return;

    setLocalStatus("checking");
    onStatusChange?.("checking");

    try {
      const response = await fetch(HEALTH_URL, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        setLocalStatus("healthy");
        onStatusChange?.("healthy");
      } else {
        setLocalStatus("unhealthy");
        onStatusChange?.("unhealthy");
      }
    } catch (error) {
      setLocalStatus("stopped");
      onStatusChange?.("stopped");
      console.error("Hive health check failed:", error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Hive</CardTitle>
        <CardDescription>Worker Manager</CardDescription>
        <CardAction>
          <Badge
            variant={statusConfig.variant}
            className="cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleHealthCheck}
          >
            {statusConfig.label}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {/* SSH Target Selection */}
          <label
            htmlFor="ssh-target"
            className="text-sm font-medium text-foreground"
          >
            SSH Target
          </label>
          <SshHivesDataProvider
            fallback={
              <Select disabled>
                <SelectTrigger id="ssh-target" className="w-full">
                  <SelectValue placeholder="Loading targets..." />
                </SelectTrigger>
              </Select>
            }
          >
            {(hives) => (
              <Select value={selectedTarget} onValueChange={setSelectedTarget}>
                <SelectTrigger id="ssh-target" className="w-full">
                  <SelectValue placeholder="Select target" />
                </SelectTrigger>
                <SelectContent>
                  {/* Always include localhost */}
                  <SelectItem value="localhost">
                    <SshTargetItem name="localhost" subtitle="This machine" />
                  </SelectItem>
                  {/* Dynamic SSH targets from ~/.ssh/config */}
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
            )}
          </SshHivesDataProvider>

          <ServiceActionButtons
            servicePrefix="hive"
            onCommandClick={onCommandClick}
            disabled={disabled}
          />
        </div>
      </CardContent>
    </Card>
  );
}
