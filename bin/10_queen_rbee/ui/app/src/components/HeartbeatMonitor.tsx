// Heartbeat Monitor Component
// Displays real-time worker and hive status
// TEAM-364: Updated to display ProcessStats telemetry

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
  Badge,
} from "@rbee/ui/atoms";
import { StatusKPI, PulseBadge, IconPlate } from "@rbee/ui/molecules";
import { Activity, Server, Cpu, ChevronDown, Gauge, HardDrive } from "lucide-react";

interface ProcessStats {
  pid: number;
  group: string;
  instance: string;
  cpu_pct: number;
  rss_mb: number;
  gpu_util_pct: number;
  vram_mb: number;
  total_vram_mb: number;
  model: string | null;
  uptime_s: number;
}

interface HiveData {
  hive_id: string;
  workers: ProcessStats[];
  last_update: string;
}

interface HeartbeatMonitorProps {
  workersOnline: number;
  hivesOnline: number;
  hives: HiveData[];
}

export function HeartbeatMonitor({
  workersOnline,
  hivesOnline,
  hives,
}: HeartbeatMonitorProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <IconPlate icon={<Activity />} tone="success" size="sm" shape="rounded" />
          Heartbeat Monitor
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* KPI Metrics */}
        <div className="grid grid-cols-2 gap-3">
          <StatusKPI
            icon={<Cpu />}
            color="success"
            label="Workers Online"
            value={workersOnline}
          />
          <StatusKPI
            icon={<Server />}
            color="primary"
            label="Active Hives"
            value={hivesOnline}
          />
        </div>

        {/* Hives List */}
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">Hives & Workers</h3>
          {hives.length === 0 ? (
            <p className="text-sm text-muted-foreground">No hives online</p>
          ) : (
            <div className="space-y-2">
              {hives.map((hive) => (
                <Collapsible key={hive.hive_id}>
                  <div className="border border-border rounded-lg overflow-hidden">
                    {/* Hive Header */}
                    <CollapsibleTrigger className="w-full flex items-center gap-3 p-3 hover:bg-accent transition-colors group">
                      <ChevronDown className="h-4 w-4 transition-transform group-data-[state=closed]:-rotate-90" />
                      <PulseBadge
                        text="Online"
                        variant="success"
                        size="sm"
                        animated
                      />
                      <span className="font-medium">{hive.hive_id}</span>
                      <Badge variant="secondary" className="ml-auto">
                        {hive.workers?.length || 0} workers
                      </Badge>
                    </CollapsibleTrigger>

                    {/* Workers List */}
                    <CollapsibleContent>
                      {hive.workers && hive.workers.length > 0 && (
                        <div className="px-3 pb-3 pl-10 space-y-1 bg-muted/30">
                          {hive.workers.map((worker) => (
                            <div
                              key={worker.pid}
                              className="flex items-center gap-2 p-2 text-sm bg-card rounded border"
                            >
                              <PulseBadge
                                text=""
                                variant={worker.gpu_util_pct > 0 ? "success" : "info"}
                                size="sm"
                                animated={worker.gpu_util_pct > 0}
                                className="px-0"
                              />
                              <span className="font-mono text-xs">
                                {worker.group}/{worker.instance}
                              </span>
                              {worker.model && (
                                <Badge variant="outline" className="text-xs">
                                  {worker.model}
                                </Badge>
                              )}
                              <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
                                <span className="flex items-center gap-1">
                                  <Gauge className="h-3 w-3" />
                                  GPU: {worker.gpu_util_pct.toFixed(1)}%
                                </span>
                                <span className="flex items-center gap-1">
                                  <HardDrive className="h-3 w-3" />
                                  VRAM: {worker.vram_mb}MB / {worker.total_vram_mb}MB
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </CollapsibleContent>
                  </div>
                </Collapsible>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
