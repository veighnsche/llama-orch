// Hive service card with lifecycle controls and SSH target selection
import { useState } from "react";
import {
  Card,
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
        <div className="flex flex-col sm:flex-row items-start sm:items-start justify-between gap-3 sm:gap-4">
          <div className="space-y-1.5 flex-1 min-w-0">
            <CardTitle className="text-base sm:text-lg">Hive</CardTitle>
            <CardDescription className="text-sm">
              Worker Manager
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
            Manages workers and catalogs (models, worker binaries) on this
            machine or remote SSH targets
          </p>

          {/* SSH Target Selection */}
          <div className="space-y-2">
            <label
              htmlFor="ssh-target"
              className="text-sm font-medium text-foreground"
            >
              Target
            </label>
            <Select value={selectedTarget} onValueChange={setSelectedTarget}>
              <SelectTrigger id="ssh-target">
                <SelectValue placeholder="Select target" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="localhost">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">localhost</span>
                    <span className="text-xs text-muted-foreground">
                      This machine
                    </span>
                  </div>
                </SelectItem>
                <SelectItem value="infra">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">infra</span>
                    <span className="text-xs text-muted-foreground">
                      vince@192.168.178.84:22
                    </span>
                  </div>
                </SelectItem>
                <SelectItem value="mac">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">mac</span>
                    <span className="text-xs text-muted-foreground">
                      vinceliem@192.168.178.15:22
                    </span>
                  </div>
                </SelectItem>
                <SelectItem value="workstation">
                  <div className="flex flex-col items-start">
                    <span className="font-medium">workstation</span>
                    <span className="text-xs text-muted-foreground">
                      vince@192.168.178.29:22
                    </span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

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
