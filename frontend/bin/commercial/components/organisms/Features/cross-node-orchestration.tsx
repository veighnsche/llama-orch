import { Network, GitBranch } from "lucide-react"
import { SectionContainer, IconBox } from '@/components/molecules'

export function CrossNodeOrchestration() {
  return (
    <SectionContainer
      title="Cross-Pool Orchestration"
      bgVariant="background"
      subtitle="Seamlessly orchestrate AI workloads across your entire network. One command runs inference on any machine in your pool."
    >
      <div className="max-w-5xl mx-auto space-y-8">
          {/* SSH Registry Management */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={Network} color="primary" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">Pool Registry Management</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Configure remote machines once, use them forever. rbee-keeper manages SSH connections, validates
                  connectivity, and maintains a registry of all available pools.
                </p>
              </div>
            </div>

            <div className="bg-background rounded-lg p-6 font-mono text-sm space-y-2">
              <div className="text-muted-foreground"># Add a remote machine to your pool</div>
              <div className="text-chart-3 mt-2">
                rbee-keeper setup add-node \
                <br />
                {"  "}--name workstation \
                <br />
                {"  "}--ssh-host workstation.home.arpa \
                <br />
                {"  "}--ssh-user vince \
                <br />
                {"  "}--ssh-key ~/.ssh/id_ed25519
              </div>
              <div className="text-muted-foreground mt-4"># Run inference on that machine</div>
              <div className="text-chart-3 mt-2">
                rbee-keeper infer --node workstation \
                <br />
                {"  "}--model hf:meta-llama/Llama-3.1-8B \
                <br />
                {"  "}--prompt "write a short story"
              </div>
            </div>

            <div className="mt-6 grid md:grid-cols-3 gap-4">
              <div className="bg-background rounded-lg p-4">
                <div className="text-primary font-bold mb-1">Automatic Detection</div>
                <div className="text-muted-foreground text-sm">Detects CUDA, Metal, CPU backends and device counts</div>
              </div>
              <div className="bg-background rounded-lg p-4">
                <div className="text-primary font-bold mb-1">SSH Validation</div>
                <div className="text-muted-foreground text-sm">Tests connectivity before saving to the pool registry</div>
              </div>
              <div className="bg-background rounded-lg p-4">
                <div className="text-primary font-bold mb-1">Zero Config</div>
                <div className="text-muted-foreground text-sm">No manual setup on remote nodes required</div>
              </div>
            </div>
          </div>

          {/* Automatic Worker Provisioning */}
          <div className="bg-card border border-border rounded-lg p-8">
            <div className="flex items-start gap-4 mb-6">
              <IconBox icon={GitBranch} color="chart-2" size="lg" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">Automatic Worker Provisioning</h3>
                <p className="text-muted-foreground leading-relaxed">
                  rbee automatically spawns workers on remote machines via SSH. No manual daemon management. Workers start
                  on-demand and shut down cleanly.
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="flex-shrink-0 w-40 text-sm text-muted-foreground">queen-rbee</div>
                <div className="flex-1 text-foreground text-sm">Orchestrator (runs on your main machine)</div>
              </div>
              <div className="flex items-center gap-3 pl-8">
                <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">↓ SSH</div>
                <div className="flex-1 text-muted-foreground/70 text-sm">Connects to remote pool managers</div>
              </div>
              <div className="flex items-center gap-3 pl-8">
                <div className="flex-shrink-0 w-40 text-sm text-muted-foreground">rbee-hive</div>
                <div className="flex-1 text-foreground text-sm">Pool manager (spawned on remote machine)</div>
              </div>
              <div className="flex items-center gap-3 pl-16">
                <div className="flex-shrink-0 w-32 text-sm text-muted-foreground">↓ Spawns</div>
                <div className="flex-1 text-muted-foreground/70 text-sm">Creates workers as needed</div>
              </div>
              <div className="flex items-center gap-3 pl-16">
                <div className="flex-shrink-0 w-40 text-sm text-muted-foreground">worker-rbee</div>
                <div className="flex-1 text-foreground text-sm">Worker daemon (handles inference)</div>
              </div>
            </div>
          </div>
        </div>
      </SectionContainer>
  )
}
