// TEAM-338: Queen service card - presentation component
// Pure presentation logic, receives data as props
// No data fetching - see QueenContainer.tsx for data layer

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
import { Play, Square, Download, RefreshCw, Trash2, Loader2 } from "lucide-react";
import { commands } from "@/generated/bindings";
import { useCommandStore } from "../store/commandStore";

// TEAM-338: Export the status type
export interface QueenStatus {
  isRunning: boolean;
  isInstalled: boolean;
}

// TEAM-338: Export the loading fallback
export function LoadingQueen() {
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

// TEAM-338: Export the presentation component
export interface QueenCardProps {
  status: QueenStatus;
  onRefresh: () => void;
}

export function QueenCard({ status, onRefresh }: QueenCardProps) {
  const { isExecuting, setIsExecuting } = useCommandStore();

  const handleCommand = async (commandPromise: Promise<unknown>) => {
    setIsExecuting(true);
    try {
      await commandPromise;
      // Refresh status after command completes
      onRefresh();
    } catch (error) {
      console.error("Queen command failed:", error);
    } finally {
      setIsExecuting(false);
    }
  };

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
            onClick={() => handleCommand(commands.queenStart())}
            disabled={isExecuting}
            className="w-full"
            dropdownContent={
              <>
                <DropdownMenuItem
                  onClick={() => handleCommand(commands.queenStop())}
                >
                  <Square className="mr-2 h-4 w-4 text-danger" />
                  Stop
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => handleCommand(commands.queenInstall(null))}
                >
                  <Download className="mr-2 h-4 w-4" />
                  Install
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => handleCommand(commands.queenRebuild(false))}
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Update
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() => handleCommand(commands.queenUninstall())}
                  variant="destructive"
                >
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
