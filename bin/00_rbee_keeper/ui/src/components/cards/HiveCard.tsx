// TEAM-338: Individual hive card component with lifecycle controls
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
import { useHive, useHiveActions } from "../../store/hiveQueries";
import { StatusBadge } from "../StatusBadge";
import { ServiceActionButton } from "./ServiceActionButton";
import type { SshHive } from "../../store/hiveQueries";

interface HiveCardProps {
  hiveId: string;
  title: string;
  description: string;
}

// TEAM-368: HiveCard is ONLY rendered for installed hives (filtered by InstalledHiveList)
export function HiveCard({ hiveId, title, description }: HiveCardProps) {
  const { data: hive, isLoading, error, refetch } = useHive(hiveId);
  const { start, stop, uninstall, rebuild } = useHiveActions();
  const { isExecuting } = useCommandStore();

  // TEAM-368: No need to check isInstalled - InstalledHiveList already filters!
  return (
    <QueryContainer<SshHive>
      isLoading={isLoading}
      error={error?.message ?? null}
      data={hive ?? null}
      onRetry={() => refetch()}
      metadata={{ name: `${title} Hive`, description }}
    >
      {(hive) => (
        <HiveCardContent
          hiveId={hiveId}
          title={title}
          description={description}
          hive={hive}
          isExecuting={isExecuting}
          actions={{ start, stop, uninstall, rebuild }}
          refetch={refetch}
        />
      )}
    </QueryContainer>
  );
}

// TEAM-368: HiveCard actions - NO INSTALL (use InstallHiveCard for that!)
type HiveCardActions = Omit<ReturnType<typeof useHiveActions>, 'install'>;

// TEAM-354: Inner component receives type-safe hive data
function HiveCardContent({
  hiveId,
  title,
  description,
  hive,
  isExecuting,
  actions,
  refetch,
}: {
  hiveId: string;
  title: string;
  description: string;
  hive: SshHive;
  isExecuting: boolean;
  actions: HiveCardActions;
  refetch: () => void;
}) {
  // TEAM-368: HiveCard only shows INSTALLED hives
  const isInstalled = true; // Always true - card doesn't show if not installed
  const isRunning = hive.status === "online";

  // Compute badge status
  const badgeStatus = isRunning
    ? ("running" as const)
      : ("stopped" as const);

  return (
    <Card className="w-80 h-80 max-w-sm flex flex-col">
      <CardHeader>
        <CardTitle>{title} Hive</CardTitle>
        <CardDescription>{description}</CardDescription>
        <CardAction>
          <StatusBadge status={badgeStatus} onClick={refetch} />
        </CardAction>
      </CardHeader>
      <div className="flex-1" />
      <CardContent>
        <div className="space-y-4">
          <p className="text-sm text-muted-foreground leading-relaxed">
            {description}
          </p>
          <ServiceActionButton
            serviceId={hiveId}
            isInstalled={isInstalled}
            isRunning={isRunning}
            isExecuting={isExecuting}
            actions={{
              start: (id) => actions.start(id!),
              stop: (id) => actions.stop(id!),
              // NO INSTALL - HiveCard only shows installed hives!
              rebuild: (id) => actions.rebuild(id!),
              uninstall: (id) => actions.uninstall(id!),
            }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
