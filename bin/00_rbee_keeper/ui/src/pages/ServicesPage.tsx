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
            "routes inference jobs to the right worker in the right hive. Start Queen first to enable job routing.",
        },
        {
          title: "Hive",
          description:
            "manages worker lifecycle and catalogs (models from HuggingFace, worker binaries). Start localhost hive to see local models and workers. Use SSH targets to start remote hives and access their catalogs.",
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
