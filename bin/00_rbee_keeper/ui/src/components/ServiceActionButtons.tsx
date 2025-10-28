// TEAM-295: Reusable service action buttons component
// Updated: Using SplitButton with wide Play button + dropdown for other actions

import {
  SplitButton,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@rbee/ui/atoms";
import { Play, Square, Download, RefreshCw, Trash2 } from "lucide-react";

interface ServiceActionButtonsProps {
  servicePrefix: string;
  onCommandClick: (command: string) => void;
  disabled?: boolean;
}

export function ServiceActionButtons({
  servicePrefix,
  onCommandClick,
  disabled = false,
}: ServiceActionButtonsProps) {
  return (
    <SplitButton
      variant="default"
      size="default"
      icon={<Play className="h-4 w-4" />}
      onClick={() => onCommandClick(`${servicePrefix}-start`)}
      disabled={disabled}
      className="w-full"
      dropdownContent={
        <>
          <DropdownMenuItem
            onClick={() => onCommandClick(`${servicePrefix}-stop`)}
          >
            <Square className="mr-2 h-4 w-4 text-danger" />
            Stop
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => onCommandClick(`${servicePrefix}-install`)}
          >
            <Download className="mr-2 h-4 w-4" />
            Install
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => onCommandClick(`${servicePrefix}-update`)}
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Update
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => onCommandClick(`${servicePrefix}-uninstall`)}
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
  );
}
