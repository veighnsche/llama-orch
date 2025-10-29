// TEAM-294: Keeper page - Status cards and SSH targets table
// TEAM-353: Migrated to query hooks (deleted DaemonContainer pattern)
// TEAM-350: Added LocalhostHive component (separate from SSH hives)
//
// Queen Lifecycle Operations:
// - start: Run the queen daemon
// - stop: Stop the queen daemon
// - install: Build from git repo (cargo build --release) and install to ~/.local/bin (errors if already installed)
// - update: Rebuild from source (cargo build --release)
// - uninstall: Remove binary from ~/.local/bin (errors if not installed)

import { PageContainer } from "@rbee/ui/molecules";
import { InstalledHiveList } from "@/components/InstalledHiveList";
import { InstallHiveCard } from "@/components/cards/InstallHiveCard";
import { QueenCard } from "../components/cards/QueenCard";

export default function KeeperPage() {
  return (
    <PageContainer
      title="Services"
      description="Manage Queen and Hive services"
      padding="default"
      helperText={[
        {
          title: "Queen",
          description:
            "The orchestrator brain. Tracks all workers and hives via heartbeats, schedules inference jobs, and routes requests directly to workers. Provides unified API and OpenAI compatibility. Start Queen first to enable orchestration.",
        },
        {
          title: "Hive",
          description:
            "Manages worker lifecycle (spawn/stop workers) and model downloads. Each hive runs on a machine with GPUs/CPUs. Queen sends jobs to hives to provision workers, then routes inference directly to workers (bypassing hive). Start localhost hive for local GPU/CPU, or connect to remote hives via their URLs.",
        },
      ]}
    >
      <div className="flex flex-wrap gap-6 justify-center">
        <QueenCard />
        <InstalledHiveList />
        <InstallHiveCard />
      </div>
    </PageContainer>
  );
}
