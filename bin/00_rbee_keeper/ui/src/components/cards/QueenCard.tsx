// TEAM-338: Queen service card - uses Zustand store for state management
// TEAM-353: Rewritten to use query hooks (deleted DaemonContainer pattern)
// TEAM-354: Fixed to use QueryContainer per CORRECT_ARCHITECTURE.md Pattern 4

import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@rbee/ui/atoms";
import { QueryContainer } from "../../containers/QueryContainer";
import { useCommandStore } from "../../store/commandStore";
import { useQueen, useQueenActions } from "../../store/queenStore";
import { StatusBadge } from "../StatusBadge";
import { ServiceActionButton } from "./ServiceActionButton";
import type { QueenStatus } from "../../store/queenStore";

// TEAM-354: Correct pattern - use QueryContainer per spec Pattern 4
export function QueenCard() {
  const { queen, isLoading, error, refetch } = useQueen();
  const { start, stop, install, rebuild, uninstall } = useQueenActions();
  const { isExecuting } = useCommandStore();

  return (
    <QueryContainer<QueenStatus>
      isLoading={isLoading}
      error={error}
      data={queen}
      onRetry={refetch}
      metadata={{ name: "Queen", description: "Smart API server" }}
    >
      {(queen) => <QueenCardContent queen={queen} isExecuting={isExecuting} actions={{ start, stop, install, rebuild, uninstall }} refetch={refetch} />}
    </QueryContainer>
  );
}

// TEAM-354: Inner component receives type-safe queen data
function QueenCardContent({
  queen,
  isExecuting,
  actions,
  refetch,
}: {
  queen: QueenStatus;
  isExecuting: boolean;
  actions: ReturnType<typeof useQueenActions>;
  refetch: () => void;
}) {
  const isRunning = queen.isRunning;
  const isInstalled = queen.isInstalled;

  // Compute badge status
  const badgeStatus = !isInstalled
    ? ("unknown" as const)
    : isRunning
      ? ("running" as const)
      : ("stopped" as const);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Queen</CardTitle>
        <CardDescription>Smart API server</CardDescription>
        <CardAction>
          <StatusBadge status={badgeStatus} onClick={refetch} />
        </CardAction>
      </CardHeader>
      <div className="flex-1" />
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            Job router that dispatches inference requests to workers in the
            correct hive
          </p>
          <ServiceActionButton
            serviceId="queen"
            isInstalled={isInstalled}
            isRunning={isRunning}
            isExecuting={isExecuting}
            actions={{
              start: () => actions.start(),
              stop: () => actions.stop(),
              install: () => actions.install(),
              rebuild: () => actions.rebuild(),
              uninstall: () => actions.uninstall(),
            }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
