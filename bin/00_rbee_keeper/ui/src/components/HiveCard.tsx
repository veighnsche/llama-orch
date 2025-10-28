// TEAM-338: Individual hive card component with lifecycle controls
// Reusable card for displaying a single hive with start/stop/uninstall actions

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
import { Play, Square, RefreshCw, Trash2 } from "lucide-react";
import { useCommandStore } from "../store/commandStore";
import { useSshHivesStore } from "../store/hiveStore";

interface HiveCardProps {
  hiveId: string;
  title: string;
  description: string;
}

export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  const { isExecuting } = useCommandStore();
  const { start, stop, uninstall, refreshCapabilities } = useSshHivesStore();

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>
        <SplitButton
          variant="default"
          size="default"
          icon={<Play className="h-4 w-4" />}
          onClick={() => start(hiveId)}
          disabled={isExecuting}
          className="w-full"
          dropdownContent={
            <>
              <DropdownMenuItem onClick={() => stop(hiveId)}>
                <Square className="mr-2 h-4 w-4" />
                Stop
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => refreshCapabilities(hiveId)}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => uninstall(hiveId)}
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
      </CardContent>
    </Card>
  );
}
