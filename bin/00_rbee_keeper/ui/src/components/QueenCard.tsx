// TEAM-338: Queen service card - uses Zustand store for state management
// Follows idiomatic Zustand pattern with useQueenStore hook
// Store handles command execution and global isExecuting internally

import { useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  SplitButton,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@rbee/ui/atoms";
import {
  Play,
  Square,
  Download,
  RefreshCw,
  Trash2,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { useQueenStore } from "../store/queenStore";
import { useCommandStore } from "../store/commandStore";

// TEAM-338: Re-export the status type from store
export type { QueenStatus } from "../store/queenStore";

export function QueenCard() {
  const {
    status,
    isLoading,
    error,
    fetchStatus,
    start,
    stop,
    install,
    rebuild,
    uninstall,
  } = useQueenStore();
  const { isExecuting } = useCommandStore();

  // TEAM-338: Fetch status on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // TEAM-338: Loading state
  if (isLoading && !status) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Queen</CardTitle>
          <CardDescription>Smart API server</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground leading-relaxed">
              Job router that dispatches inference requests to workers in the
              correct hive
            </p>
            <div className="flex items-center justify-center py-4">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // TEAM-338: Error state
  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Queen</CardTitle>
          <CardDescription>Smart API server</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-4 w-4" />
              <p className="text-sm">{error}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Queen</CardTitle>
        <CardDescription>Smart API server</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Job router that dispatches inference requests to workers in the
            correct hive
          </p>
          <SplitButton
            variant="default"
            size="default"
            icon={<Play className="h-4 w-4" />}
            onClick={start}
            disabled={isExecuting}
            className="w-full"
            dropdownContent={
              <>
                <DropdownMenuItem onClick={stop}>
                  <Square className="mr-2 h-4 w-4 text-danger" />
                  Stop
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={install}>
                  <Download className="mr-2 h-4 w-4" />
                  Install
                </DropdownMenuItem>
                <DropdownMenuItem onClick={rebuild}>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Update
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={uninstall} variant="destructive">
                  <Trash2 className="mr-2 h-4 w-4" />
                  Uninstall
                </DropdownMenuItem>
              </>
            }
          >
            Start
          </SplitButton>
        </div>
      </CardContent>
    </Card>
  );
}
