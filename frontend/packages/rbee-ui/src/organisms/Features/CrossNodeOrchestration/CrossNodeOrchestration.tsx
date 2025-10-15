import { Badge } from '@rbee/ui/atoms/Badge'
import { Separator } from '@rbee/ui/atoms/Separator'
import { IconPlate, SectionContainer } from '@rbee/ui/molecules'
import { ArrowDown, Badge as BadgeIcon, GitBranch, Network } from 'lucide-react'

export function CrossNodeOrchestration() {
  return (
    <SectionContainer
      title="Cross-Pool Orchestration"
      bgVariant="background"
      subtitle="Seamlessly orchestrate AI workloads across your entire network. One command runs inference on any machine in your pool."
    >
      <div className="max-w-6xl mx-auto">
        {/* Overline badge */}
        <div className="flex justify-center mb-8">
          <Badge variant="secondary">Distributed execution</Badge>
        </div>

        <div className="grid gap-8 lg:grid-cols-2 items-start">
          {/* Pool Registry Management Card */}
          <div className="bg-card border rounded-2xl p-8 space-y-6 animate-in fade-in slide-in-from-left-4 duration-500">
            <div className="flex items-start gap-4">
              <IconPlate icon={Network} tone="primary" size="md" shape="rounded" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">Pool Registry Management</h3>
                <p className="text-muted-foreground leading-relaxed">
                  Configure remote machines once. rbee-keeper handles SSH, validates connectivity, and keeps your pool
                  registry synced.
                </p>
              </div>
            </div>

            <div
              className="bg-background rounded-lg p-6 font-mono text-sm leading-relaxed"
              aria-label="Pool registry CLI example"
            >
              <div className="text-muted-foreground"># Add a remote machine to your pool</div>
              <div className="text-chart-3 mt-2">
                $ rbee-keeper setup add-node \
                <br />
                {'  '}--name workstation \
                <br />
                {'  '}--ssh-host workstation.home.arpa \
                <br />
                {'  '}--ssh-user vince \
                <br />
                {'  '}--ssh-key ~/.ssh/id_ed25519
              </div>
              <div className="text-muted-foreground mt-4"># Run inference on that machine</div>
              <div className="text-chart-3 mt-2">
                $ rbee-keeper infer --node workstation \
                <br />
                {'  '}--model hf:meta-llama/Llama-3.1-8B \
                <br />
                {'  '}--prompt &quot;write a short story&quot;
              </div>
            </div>

            <div className="grid sm:grid-cols-3 gap-3">
              <div className="bg-background rounded-lg p-4 transition-transform hover:-translate-y-0.5">
                <div className="text-sm font-semibold text-foreground mb-1">Automatic Detection</div>
                <div className="text-xs text-muted-foreground">Detects CUDA, Metal, CPU & device counts.</div>
              </div>
              <div className="bg-background rounded-lg p-4 transition-transform hover:-translate-y-0.5">
                <div className="text-sm font-semibold text-foreground mb-1">SSH Validation</div>
                <div className="text-xs text-muted-foreground">Connectivity tested before save.</div>
              </div>
              <div className="bg-background rounded-lg p-4 transition-transform hover:-translate-y-0.5">
                <div className="text-sm font-semibold text-foreground mb-1">Zero Config</div>
                <div className="text-xs text-muted-foreground">No manual setup on remote nodes.</div>
              </div>
            </div>
          </div>

          <Separator className="lg:hidden my-2 opacity-40" />

          {/* Automatic Worker Provisioning Card */}
          <div className="bg-card border rounded-2xl p-8 space-y-6 animate-in fade-in slide-in-from-right-4 duration-500 delay-100">
            <div className="flex items-start gap-4">
              <IconPlate icon={GitBranch} tone="chart-2" size="md" shape="rounded" className="flex-shrink-0" />
              <div>
                <h3 className="text-2xl font-bold tracking-tight text-foreground mb-2">
                  Automatic Worker Provisioning
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  rbee spawns workers via SSH on demand and shuts them down cleanly. No manual daemons.
                </p>
              </div>
            </div>

            {/* Diagram */}
            <div className="relative bg-background rounded-xl p-6 overflow-hidden">
              <div className="space-y-4">
                {/* Row 1: queen-rbee */}
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="inline-flex items-center gap-2 bg-primary/10 border border-primary/20 rounded-lg px-4 py-2">
                      <span className="font-mono text-sm font-semibold text-foreground">queen-rbee</span>
                      <Badge variant="secondary" className="text-xs">
                        Orchestrator
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Arrow SSH */}
                <div className="flex items-center gap-2 pl-8">
                  <ArrowDown className="size-4 text-muted-foreground" aria-hidden="true" />
                  <span className="text-xs text-muted-foreground font-medium">SSH</span>
                </div>

                {/* Row 2: rbee-hive */}
                <div className="flex items-center gap-3 pl-8">
                  <div className="flex-1">
                    <div className="inline-flex items-center gap-2 bg-chart-2/10 border border-chart-2/20 rounded-lg px-4 py-2">
                      <span className="font-mono text-sm font-semibold text-foreground">rbee-hive</span>
                      <Badge variant="secondary" className="text-xs">
                        Pool manager
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Arrow Spawns */}
                <div className="flex items-center gap-2 pl-16">
                  <ArrowDown className="size-4 text-muted-foreground" aria-hidden="true" />
                  <span className="text-xs text-muted-foreground font-medium">Spawns</span>
                </div>

                {/* Row 3: worker-rbee */}
                <div className="flex items-center gap-3 pl-16">
                  <div className="flex-1">
                    <div className="inline-flex items-center gap-2 bg-chart-3/10 border border-chart-3/20 rounded-lg px-4 py-2">
                      <span className="font-mono text-sm font-semibold text-foreground">worker-rbee</span>
                      <Badge variant="secondary" className="text-xs">
                        Inference worker
                      </Badge>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Legend */}
            <div className="grid sm:grid-cols-3 gap-3 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="size-2 rounded-full bg-chart-3 shrink-0" aria-hidden="true" />
                <span>On-demand start</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="size-2 rounded-full bg-chart-3 shrink-0" aria-hidden="true" />
                <span>Clean shutdown</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="size-2 rounded-full bg-chart-3 shrink-0" aria-hidden="true" />
                <span>No daemon drift</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
