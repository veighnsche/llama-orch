// Queen service card with lifecycle controls
import { useState } from "react";
import {
  Card,
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

interface QueenCardProps {
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

const HEALTH_URL = "http://localhost:7833/health";

export function QueenCard({
  status = "unknown",
  onCommandClick,
  onStatusChange,
  disabled = false,
}: QueenCardProps) {
  const [localStatus, setLocalStatus] = useState<ServiceStatus>(status);
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
      console.error("Queen health check failed:", error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col sm:flex-row items-start sm:items-start justify-between gap-3 sm:gap-4">
          <div className="space-y-1.5 flex-1 min-w-0">
            <CardTitle className="text-base sm:text-lg">Queen</CardTitle>
            <CardDescription className="text-sm">
              Smart API server
            </CardDescription>
          </div>
          <Badge
            variant={statusConfig.variant}
            className="cursor-pointer hover:opacity-80 transition-opacity shrink-0 self-start"
            onClick={handleHealthCheck}
          >
            {statusConfig.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Job router that dispatches inference requests to workers in the
            correct hive
          </p>
          <ServiceActionButtons
            servicePrefix="queen"
            onCommandClick={onCommandClick}
            disabled={disabled}
          />
        </div>
      </CardContent>
    </Card>
  );
}
