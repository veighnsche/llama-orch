// TEAM-295: Reusable service action buttons component

import {
  Button,
  Tooltip,
  TooltipTrigger,
  TooltipContent,
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
    <div className="flex flex-wrap justify-around gap-2">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={() => onCommandClick(`${servicePrefix}-start`)}
            disabled={disabled}
          >
            <Play className="h-4 w-4 text-success" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Start</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={() => onCommandClick(`${servicePrefix}-stop`)}
            disabled={disabled}
          >
            <Square className="h-4 w-4 text-danger" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Stop</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={() => onCommandClick(`${servicePrefix}-install`)}
            disabled={disabled}
          >
            <Download className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Install</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={() => onCommandClick(`${servicePrefix}-update`)}
            disabled={disabled}
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Update</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            size="icon-sm"
            variant="ghost"
            onClick={() => onCommandClick(`${servicePrefix}-uninstall`)}
            disabled={disabled}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Uninstall</TooltipContent>
      </Tooltip>
    </div>
  );
}
