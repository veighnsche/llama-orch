// Heartbeat Monitor Component
// Displays real-time worker and hive status

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
import { Activity, Server, Cpu, ChevronDown } from "lucide-react";

interface HeartbeatMonitorProps {
  workersOnline: number;
  hives: any[];
}

export function HeartbeatMonitor({
  workersOnline,
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
            value={hives.length}
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
                <Collapsible key={hive.id}>
                  <div className="border rounded-lg overflow-hidden">
                    {/* Hive Header */}
                    <CollapsibleTrigger className="w-full flex items-center gap-3 p-3 hover:bg-accent transition-colors group">
                      <ChevronDown className="h-4 w-4 transition-transform group-data-[state=closed]:-rotate-90" />
                      <PulseBadge
                        text="Online"
                        variant="success"
                        size="sm"
                        animated
                      />
                      <span className="font-medium">{hive.id}</span>
                      <Badge variant="secondary" className="ml-auto">
                        {hive.workers?.length || 0} workers
                      </Badge>
                    </CollapsibleTrigger>

                    {/* Workers List */}
                    <CollapsibleContent>
                      {hive.workers && hive.workers.length > 0 && (
                        <div className="px-3 pb-3 pl-10 space-y-1 bg-muted/30">
                          {hive.workers.map((worker: any) => (
                            <div
                              key={worker.id}
                              className="flex items-center gap-3 p-2 text-sm bg-card rounded border"
                            >
                              <PulseBadge
                                text=""
                                variant="success"
                                size="sm"
                                animated
                                className="px-0"
                              />
                              <span className="font-mono text-xs">
                                {worker.id}
                              </span>
                              <Badge variant="outline" className="ml-auto text-xs">
                                {worker.model_id}
                              </Badge>
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
