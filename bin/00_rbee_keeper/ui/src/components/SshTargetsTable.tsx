// TEAM-294: SSH Targets table component
// Displays SSH hosts from ~/.ssh/config with actions dropdown

import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  Button,
} from "@rbee/ui/atoms";
import {
  MoreVertical,
  Terminal,
  Trash2,
  RefreshCw,
  Play,
  Square,
} from "lucide-react";
import { StatusBadge } from "./StatusBadge";

interface CommandResponse {
  success: boolean;
  message: string;
  data?: string;
}

interface SSHTarget {
  host: string;
  host_subtitle?: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
}

export function SshTargetsTable() {
  const [sshTargets, setSshTargets] = useState<SSHTarget[]>([]);
  const [isLoadingTargets, setIsLoadingTargets] = useState(true);

  useEffect(() => {
    loadTargets();
  }, []);

  async function loadTargets() {
    setIsLoadingTargets(true);

    try {
      const result = await invoke<string>("hive_list");
      const response: CommandResponse = JSON.parse(result);

      if (response.success && response.data) {
        const targets: SSHTarget[] = JSON.parse(response.data);
        setSshTargets(targets);
      }
    } catch (error) {
      console.error("Failed to load SSH targets:", error);
    } finally {
      setIsLoadingTargets(false);
    }
  }

  return (
    <div className="rounded-lg border border-border bg-card">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Host</TableHead>
            <TableHead>Connection</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">
              <Button
                onClick={loadTargets}
                disabled={isLoadingTargets}
                variant="ghost"
                size="icon-sm"
                aria-label="Refresh"
                title="Refresh SSH targets"
              >
                <RefreshCw className={isLoadingTargets ? "animate-spin" : ""} />
              </Button>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {isLoadingTargets ? (
            <TableRow>
              <TableCell
                colSpan={4}
                className="text-center text-muted-foreground"
              >
                Loading SSH targets...
              </TableCell>
            </TableRow>
          ) : sshTargets.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={4}
                className="text-center text-muted-foreground"
              >
                No SSH hosts found in ~/.ssh/config
              </TableCell>
            </TableRow>
          ) : (
            sshTargets.map((target) => (
              <TableRow key={target.host}>
                <TableCell className="font-medium">
                  <div className="flex flex-col">
                    <span>{target.host}</span>
                    {target.host_subtitle && (
                      <span className="text-xs text-muted-foreground font-normal">
                        {target.host_subtitle}
                      </span>
                    )}
                  </div>
                </TableCell>
                <TableCell className="font-mono text-xs">
                  <div className="flex flex-col">
                    <span className="font-medium">{target.user}@</span>
                    <span className="text-muted-foreground font-normal">
                      {target.hostname}:{target.port}
                    </span>
                  </div>
                </TableCell>
                <TableCell>
                  <StatusBadge status={target.status} />
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-1 justify-end">
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Start hive"
                      title="Start hive"
                      className="text-success hover:text-success hover:bg-success-muted"
                    >
                      <Play className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Stop hive"
                      title="Stop hive"
                      className="text-danger hover:text-danger hover:bg-danger-muted"
                    >
                      <Square className="h-4 w-4" />
                    </Button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          aria-label="Actions"
                        >
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem>
                          <Terminal className="mr-2 h-4 w-4" />
                          Connect SSH
                        </DropdownMenuItem>
                        <DropdownMenuItem>
                          <RefreshCw className="mr-2 h-4 w-4" />
                          Check Status
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem variant="destructive">
                          <Trash2 className="mr-2 h-4 w-4" />
                          Remove
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
