// TEAM-338: Individual hive card component with lifecycle controls
// Reusable card for displaying a single hive with start/stop/uninstall actions
// TEAM-338: Made status-aware like QueenCard (StatusBadge + conditional actions)

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  SplitButton,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@rbee/ui/atoms";
import { Play, Square, RefreshCw, Trash2, Download } from "lucide-react";
import { useCommandStore } from "../store/commandStore";
import { useSshHivesStore } from "../store/hiveStore";
import { StatusBadge } from "./StatusBadge";

interface HiveCardProps {
  hiveId: string;
  title: string;
  description: string;
}

export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  const { isExecuting } = useCommandStore();
  const {
    hives,
    installedHives,
    isLoading,
    start,
    stop,
    install,
    uninstall,
    refreshCapabilities,
    fetchHiveStatus,
  } = useSshHivesStore();

  // TEAM-338: Find hive status from store (single source of truth)
  const hive = hives.find((h) => h.host === hiveId);
  const isInstalled = hive?.isInstalled ?? installedHives.includes(hiveId);
  const isRunning = hive?.status === "online";

  // TEAM-338: Compute UI state based on status (same pattern as QueenCard)
  const uiState = !isInstalled
    ? {
        mainAction: () => install(hiveId),
        mainIcon: <Download className="h-4 w-4" />,
        mainLabel: "Install",
        mainVariant: "default" as const,
        badgeStatus: "unknown" as const,
      }
    : isRunning
      ? {
          mainAction: () => stop(hiveId),
          mainIcon: <Square className="h-4 w-4" />,
          mainLabel: "Stop",
          mainVariant: "destructive" as const,
          badgeStatus: "running" as const,
        }
      : {
          mainAction: () => start(hiveId),
          mainIcon: <Play className="h-4 w-4" />,
          mainLabel: "Start",
          mainVariant: "default" as const,
          badgeStatus: "stopped" as const,
        };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction>
          <StatusBadge
            status={uiState.badgeStatus}
            onClick={() => fetchHiveStatus(hiveId)}
            isLoading={isLoading}
          />
        </CardAction>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            {description}
          </p>
          <SplitButton
            variant={uiState.mainVariant}
            size="default"
            icon={uiState.mainIcon}
            onClick={uiState.mainAction}
            disabled={isExecuting}
            className="w-full"
            dropdownContent={
              <>
                {/* Show Start if installed and not running */}
                {isInstalled && !isRunning && (
                  <>
                    <DropdownMenuItem onClick={() => start(hiveId)}>
                      <Play className="mr-2 h-4 w-4" />
                      Start
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}

                {/* Show Stop if running */}
                {isRunning && (
                  <>
                    <DropdownMenuItem onClick={() => stop(hiveId)} variant="destructive">
                      <Square className="mr-2 h-4 w-4" />
                      Stop
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}

                {/* Show Install only if not installed */}
                {!isInstalled && (
                  <DropdownMenuItem onClick={() => install(hiveId)}>
                    <Download className="mr-2 h-4 w-4" />
                    Install
                  </DropdownMenuItem>
                )}

                {/* Show Refresh only if installed */}
                {isInstalled && (
                  <DropdownMenuItem onClick={() => refreshCapabilities(hiveId)}>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Refresh
                  </DropdownMenuItem>
                )}

                {/* Show Uninstall only if installed */}
                {isInstalled && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => uninstall(hiveId)} variant="destructive">
                      <Trash2 className="mr-2 h-4 w-4" />
                      Uninstall
                    </DropdownMenuItem>
                  </>
                )}
              </>
            }
          >
            {uiState.mainLabel}
          </SplitButton>
        </div>
      </CardContent>
    </Card>
  );
}
