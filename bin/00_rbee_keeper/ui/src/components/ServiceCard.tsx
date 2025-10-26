// TEAM-296: Reusable service card component for Queen and Hive
// TEAM-296: Added status badge support (healthy, unhealthy, stopped, not-installed, out-of-date)
// TEAM-296: Added health check on badge click

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

interface ServiceCardProps {
  title: string;
  description: string;
  details: string;
  servicePrefix: "queen" | "hive";
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

// Health check URLs for each service
const HEALTH_URLS: Record<"queen" | "hive", string> = {
  queen: "http://localhost:7833/health",
  hive: "http://localhost:7835/health",
};

export function ServiceCard({
  title,
  description,
  details,
  servicePrefix,
  status = "unknown",
  onCommandClick,
  onStatusChange,
  disabled = false,
}: ServiceCardProps) {
  const [localStatus, setLocalStatus] = useState<ServiceStatus>(status);
  const currentStatus = status !== "unknown" ? status : localStatus;
  const statusConfig = STATUS_CONFIG[currentStatus];

  const handleHealthCheck = async () => {
    // Don't check if already checking
    if (currentStatus === "checking") return;

    setLocalStatus("checking");
    onStatusChange?.("checking");

    try {
      const healthUrl = HEALTH_URLS[servicePrefix];
      const response = await fetch(healthUrl, {
        method: "GET",
        signal: AbortSignal.timeout(5000), // 5 second timeout
      });

      if (response.ok) {
        setLocalStatus("healthy");
        onStatusChange?.("healthy");
      } else {
        setLocalStatus("unhealthy");
        onStatusChange?.("unhealthy");
      }
    } catch (error) {
      // Network error or timeout = service is stopped/unreachable
      setLocalStatus("stopped");
      onStatusChange?.("stopped");
      console.error(`Health check failed for ${servicePrefix}:`, error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1.5">
            <CardTitle>{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <Badge
            variant={statusConfig.variant}
            className="cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleHealthCheck}
          >
            {statusConfig.label}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground">{details}</p>
          <ServiceActionButtons
            servicePrefix={servicePrefix}
            onCommandClick={onCommandClick}
            disabled={disabled}
          />
        </div>
      </CardContent>
    </Card>
  );
}
