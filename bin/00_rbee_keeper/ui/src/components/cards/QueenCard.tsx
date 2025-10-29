// TEAM-338: Queen service card - uses Zustand store for state management
// Follows idiomatic Zustand pattern with useQueenStore hook
// Store handles command execution and global isExecuting internally
// TEAM-338: Loading/error states moved to QueenContainer (React 19 Suspense pattern)
// TEAM-340: Self-contained component with DaemonContainer wrapper (Rule Zero)

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  DropdownMenuItem,
  DropdownMenuSeparator,
  SplitButton,
} from "@rbee/ui/atoms";
import { Download, Play, RefreshCw, Square, Trash2 } from "lucide-react";
import { DaemonContainer } from "../../containers/DaemonContainer";
import { useCommandStore } from "../../store/commandStore";
import { useQueenStore } from "../../store/queenStore";
import { StatusBadge } from "../StatusBadge";

// TEAM-338: Re-export the status type from store
export type { QueenStatus } from "../../store/queenStore";

// TEAM-340: Inner component that renders after data is loaded
function QueenCardContent() {
  const {
    status,
    isLoading,
    fetchStatus,
    start,
    stop,
    install,
    rebuild,
    uninstall,
  } = useQueenStore();
  const { isExecuting } = useCommandStore();

  // TEAM-338: Compute UI state based on status (single source of truth)
  const isRunning = status?.isRunning ?? false;
  const isInstalled = status?.isInstalled ?? false;

  const uiState = !isInstalled
    ? {
        mainAction: install,
        mainIcon: <Download className="h-4 w-4" />,
        mainLabel: "Install",
        mainVariant: "default" as const,
        badgeStatus: "unknown" as const,
      }
    : isRunning
      ? {
          mainAction: stop,
          mainIcon: <Square className="h-4 w-4" />,
          mainLabel: "Stop",
          mainVariant: "destructive" as const,
          badgeStatus: "running" as const,
        }
      : {
          mainAction: start,
          mainIcon: <Play className="h-4 w-4" />,
          mainLabel: "Start",
          mainVariant: "default" as const,
          badgeStatus: "stopped" as const,
        };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Queen</CardTitle>
        <CardDescription>Smart API server</CardDescription>
        <CardAction>
          <StatusBadge
            status={uiState.badgeStatus}
            onClick={fetchStatus}
            isLoading={isLoading}
          />
        </CardAction>
      </CardHeader>
      <div className="flex-1" />
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Job router that dispatches inference requests to workers in the
            correct hive
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
                    <DropdownMenuItem onClick={start}>
                      <Play className="mr-2 h-4 w-4" />
                      Start
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}

                {/* Show Stop if running */}
                {isRunning && (
                  <>
                    <DropdownMenuItem onClick={stop} variant="destructive">
                      <Square className="mr-2 h-4 w-4" />
                      Stop
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                  </>
                )}

                {/* Show Install only if not installed */}
                {!isInstalled && (
                  <DropdownMenuItem onClick={install}>
                    <Download className="mr-2 h-4 w-4" />
                    Install
                  </DropdownMenuItem>
                )}

                {/* Show Update only if installed */}
                {isInstalled && (
                  <DropdownMenuItem onClick={rebuild}>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Update
                  </DropdownMenuItem>
                )}

                {/* Show Uninstall only if installed */}
                {isInstalled && (
                  <>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={uninstall} variant="destructive">
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

// TEAM-340: Self-contained component with DaemonContainer wrapper
export function QueenCard() {
  return (
    <DaemonContainer
      cacheKey="queen"
      metadata={{
        name: "Queen",
        description: "Smart API server",
      }}
      fetchFn={() => useQueenStore.getState().fetchStatus()}
    >
      <QueenCardContent />
    </DaemonContainer>
  );
}
