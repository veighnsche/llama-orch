// TEAM-357: Unified service action button for Queen, Localhost, and SSH Hives
// Single component with consistent logic for all services

import {
  DropdownMenuItem,
  DropdownMenuSeparator,
  SplitButton,
} from "@rbee/ui/atoms";
import { Download, Play, RefreshCw, Square, Trash2 } from "lucide-react";

interface ServiceActionButtonProps {
  serviceId: string; // "localhost", hiveId, or "queen"
  isInstalled: boolean;
  isRunning: boolean;
  isExecuting: boolean;
  actions: {
    start: (id?: string) => Promise<void>;
    stop: (id?: string) => Promise<void>;
    install?: (id?: string) => Promise<void>;
    installProd?: (id?: string) => Promise<void>; // Production install
    rebuild?: (id?: string) => Promise<void>;
    uninstall?: (id?: string) => Promise<void>;
  };
  className?: string;
}

export function ServiceActionButton({
  serviceId,
  isInstalled,
  isRunning,
  isExecuting,
  actions,
  className = "w-full",
}: ServiceActionButtonProps) {
  const { start, stop, install, installProd, rebuild, uninstall } = actions;

  // Compute main action based on status
  const mainAction = !isInstalled
    ? {
        action: () => install?.(serviceId),
        icon: <Download className="h-4 w-4" />,
        label: "Install",
        variant: "default" as const,
        disabled: !install,
      }
    : isRunning
      ? {
          action: () => stop(serviceId),
          icon: <Square className="h-4 w-4" />,
          label: "Stop",
          variant: "destructive" as const,
          disabled: false,
        }
      : {
          action: () => start(serviceId),
          icon: <Play className="h-4 w-4" />,
          label: "Start",
          variant: "default" as const,
          disabled: false,
        };

  return (
    <SplitButton
      variant={mainAction.variant}
      size="default"
      icon={mainAction.icon}
      onClick={mainAction.action}
      disabled={isExecuting || mainAction.disabled}
      className={className}
      dropdownContent={
        <>
          {/* Install (Production) - only when not installed */}
          {!isInstalled && installProd && (
            <>
              <DropdownMenuItem onClick={() => installProd(serviceId)}>
                <Download className="mr-2 h-4 w-4" />
                Install (Production)
              </DropdownMenuItem>
              <DropdownMenuSeparator />
            </>
          )}

          {/* Rebuild - available when installed (works even when running) */}
          {isInstalled && rebuild && (
            <>
              <DropdownMenuItem onClick={() => rebuild(serviceId)}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Rebuild
              </DropdownMenuItem>
              <DropdownMenuSeparator />
            </>
          )}

          {/* Uninstall - only when stopped */}
          {isInstalled && !isRunning && uninstall && (
            <DropdownMenuItem
              onClick={() => uninstall(serviceId)}
              variant="destructive"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Uninstall
            </DropdownMenuItem>
          )}
        </>
      }
    >
      {mainAction.label}
    </SplitButton>
  );
}
