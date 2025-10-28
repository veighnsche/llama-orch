// Card for managing an installed hive instance
import { useState } from "react";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Badge,
} from "@rbee/ui/atoms";
import { ServiceActionButtons } from "./ServiceActionButtons";

export type ServiceStatus =
  | "healthy"
  | "unhealthy"
  | "stopped"
  | "not-installed"
  | "out-of-date"
  | "unknown"
  | "checking";

interface InstalledHiveCardProps {
  targetId: string;
  targetName: string;
  targetSubtitle: string;
  status?: ServiceStatus;
  onCommandClick: (command: string, targetId: string) => void;
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

export function InstalledHiveCard({
  targetId,
  targetName,
  targetSubtitle,
  status = "unknown",
  onCommandClick,
  onStatusChange,
  disabled = false,
}: InstalledHiveCardProps) {
  const [localStatus, setLocalStatus] = useState<ServiceStatus>(status);
  const currentStatus = status !== "unknown" ? status : localStatus;
  const statusConfig = STATUS_CONFIG[currentStatus];

  const handleHealthCheck = async () => {
    if (currentStatus === "checking") return;

    setLocalStatus("checking");
    onStatusChange?.("checking");

    try {
      // TODO: Implement health check for remote hives
      // For now, just set to unknown
      setLocalStatus("unknown");
      onStatusChange?.("unknown");
    } catch (error) {
      setLocalStatus("stopped");
      onStatusChange?.("stopped");
      console.error(`Hive health check failed for ${targetId}:`, error);
    }
  };

  const handleCommand = (command: string) => {
    onCommandClick(command, targetId);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{targetName}</CardTitle>
        <CardDescription>{targetSubtitle}</CardDescription>
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
        <ServiceActionButtons
          servicePrefix="hive"
          onCommandClick={handleCommand}
          disabled={disabled}
        />
      </CardContent>
    </Card>
  );
}
