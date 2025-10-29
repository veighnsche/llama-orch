// Heartbeat Monitor Component
// Displays real-time worker and hive status

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@rbee/ui/atoms";
import { ChevronDown, ChevronRight, Circle } from "lucide-react";

interface HeartbeatMonitorProps {
  workersOnline: number;
  hives: any[];
}

export function HeartbeatMonitor({ workersOnline, hives }: HeartbeatMonitorProps) {
  const [expandedHives, setExpandedHives] = useState<Set<string>>(new Set());

  const toggleHive = (hiveId: string) => {
    const newExpanded = new Set(expandedHives);
    if (newExpanded.has(hiveId)) {
      newExpanded.delete(hiveId);
    } else {
      newExpanded.add(hiveId);
    }
    setExpandedHives(newExpanded);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Circle className="h-5 w-5 text-green-500 fill-green-500" />
          Heartbeat Monitor
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Workers Online:</span>
            <span className="font-bold">{workersOnline}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Hives:</span>
            <span className="font-bold">{hives.length}</span>
          </div>
        </div>

        <div className="mt-6 space-y-2">
          <h3 className="text-sm font-semibold mb-3">Hives & Workers</h3>
          {hives.length === 0 ? (
            <p className="text-sm text-muted-foreground">No hives online</p>
          ) : (
            <div className="space-y-1">
              {hives.map((hive) => (
                <div key={hive.id} className="border rounded-lg">
                  {/* Hive Header */}
                  <button
                    onClick={() => toggleHive(hive.id)}
                    className="w-full flex items-center gap-2 p-3 hover:bg-accent rounded-lg transition-colors"
                  >
                    {expandedHives.has(hive.id) ? (
                      <ChevronDown className="h-4 w-4" />
                    ) : (
                      <ChevronRight className="h-4 w-4" />
                    )}
                    <Circle className="h-3 w-3 text-blue-500 fill-blue-500" />
                    <span className="font-medium">{hive.id}</span>
                    <span className="ml-auto text-xs text-muted-foreground">
                      {hive.workers?.length || 0} workers
                    </span>
                  </button>

                  {/* Workers List */}
                  {expandedHives.has(hive.id) &&
                    hive.workers &&
                    hive.workers.length > 0 && (
                      <div className="px-3 pb-3 pl-10 space-y-1">
                        {hive.workers.map((worker: any) => (
                          <div
                            key={worker.id}
                            className="flex items-center gap-2 p-2 text-sm bg-muted/50 rounded"
                          >
                            <Circle className="h-2 w-2 text-green-500 fill-green-500" />
                            <span className="font-mono text-xs">
                              {worker.id}
                            </span>
                            <span className="ml-auto text-xs text-muted-foreground">
                              {worker.model_id}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
