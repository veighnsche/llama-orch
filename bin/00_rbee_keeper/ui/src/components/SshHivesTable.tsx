// TEAM-296: SSH Hives table presentation component
// Pure presentation component that renders the hives table

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

// TEAM-296: Loading fallback component
export function LoadingHives() {
  return (
    <div className="rounded-lg border border-border bg-card">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Host</TableHead>
            <TableHead>Connection</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">
              <Button variant="ghost" size="icon-sm" disabled>
                <RefreshCw className="animate-spin" />
              </Button>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell colSpan={4} className="text-center text-muted-foreground">
              Loading SSH hives...
            </TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </div>
  );
}

// TEAM-296: Hive data type
export interface SshHive {
  host: string;
  host_subtitle?: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
}

export interface SshHivesTableProps {
  hives: SshHive[];
  onRefresh: () => void;
}

// TEAM-296: Presentation component - renders the hives table
export function SshHivesTable({ hives, onRefresh }: SshHivesTableProps) {
  return (
    <div className="rounded-lg border border-border bg-card">
      <div className="overflow-x-auto overflow-y-visible">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="min-w-[120px]">Host</TableHead>
              <TableHead className="min-w-[150px]">Connection</TableHead>
              <TableHead className="min-w-[100px]">Status</TableHead>
              <TableHead className="text-right min-w-[140px]">
                <Button
                  onClick={onRefresh}
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Refresh"
                  title="Refresh SSH hives"
                >
                  <RefreshCw />
                </Button>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {hives.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={4}
                  className="text-center text-muted-foreground py-8"
                >
                  No SSH hives found in ~/.ssh/config
                </TableCell>
              </TableRow>
            ) : (
              hives.map((hive) => (
                <TableRow key={hive.host}>
                  <TableCell className="font-medium">
                    <div className="flex flex-col gap-0.5">
                      <span className="text-sm sm:text-base">{hive.host}</span>
                      {hive.host_subtitle && (
                        <span className="text-xs text-muted-foreground font-normal">
                          {hive.host_subtitle}
                        </span>
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="font-mono text-xs">
                    <div className="flex flex-col gap-0.5">
                      <span className="font-medium">{hive.user}@</span>
                      <span className="text-muted-foreground font-normal break-all">
                        {hive.hostname}:{hive.port}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={hive.status} />
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1 justify-end flex-nowrap">
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Start hive"
                        title="Start hive"
                        className="text-success hover:text-success hover:bg-success-muted shrink-0"
                      >
                        <Play className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label="Stop hive"
                        title="Stop hive"
                        className="text-danger hover:text-danger hover:bg-danger-muted shrink-0"
                      >
                        <Square className="h-4 w-4" />
                      </Button>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon-sm"
                            aria-label="Actions"
                            className="shrink-0"
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
    </div>
  );
}
