// TEAM-294: Keeper page - Status cards and SSH targets table
// TEAM-295: Added action buttons to Queen and Hive cards (icon-only with tooltips)
// TEAM-296: Wired up queen lifecycle operations (install, update, uninstall)
// Updated: Replaced ServiceCard with dedicated QueenCard and HiveCard components
//
// Queen Lifecycle Operations:
// - start: Run the queen daemon
// - stop: Stop the queen daemon
// - install: Build from git repo (cargo build --release) and install to ~/.local/bin (errors if already installed)
// - update: Rebuild from source (cargo build --release)
// - uninstall: Remove binary from ~/.local/bin (errors if not installed)

import { PageContainer } from "@rbee/ui/molecules";
import { QueenDataProvider } from "../containers/QueenContainer";
import { QueenCard, LoadingQueen } from "../components/QueenCard";

export default function KeeperPage() {
  return (
    <PageContainer
      title="Services"
      description="Manage Queen and Hive services"
      padding="lg"
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
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        <QueenDataProvider fallback={<LoadingQueen />}>
          {(status, onRefresh) => (
            <QueenCard status={status} onRefresh={onRefresh} />
          )}
        </QueenDataProvider>
      </div>
    </PageContainer>
  );
}
