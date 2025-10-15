import { Badge } from '@rbee/ui/atoms/Badge'
import { FeatureCard, SectionContainer } from '@rbee/ui/molecules'
import { ChevronRight, Code, Database, Network, Shield, Terminal } from 'lucide-react'

export function AdditionalFeaturesGrid() {
  return (
    <SectionContainer title="Everything You Need for AI Infrastructure" bgVariant="background">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Overline badge */}
        <div className="flex justify-center">
          <Badge variant="secondary">Capabilities overview</Badge>
        </div>

        {/* Row 1: Core Platform */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Badge variant="secondary">Core Platform</Badge>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Cascading Shutdown */}
            <a
              href="#security-isolation"
              className="group relative rounded-2xl border bg-card p-6 transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-chart-2"
              aria-label="Learn more about Cascading Shutdown"
            >
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-chart-2/10 p-3">
                  <Shield className="size-6 text-chart-2" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-foreground mb-2">Cascading Shutdown</h3>
                  <p className="text-sm text-muted-foreground">
                    Ctrl+C tears down keeper → queen → hive → workers. No orphans, no VRAM leaks.
                  </p>
                </div>
              </div>
              <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                <span>Learn more</span>
                <ChevronRight className="size-3" />
              </div>
            </a>

            {/* Model Catalog */}
            <a
              href="#intelligent-model-management"
              className="group relative rounded-2xl border bg-card p-6 transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-100 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-chart-3"
              aria-label="Learn more about Model Catalog"
            >
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-chart-3/10 p-3">
                  <Database className="size-6 text-chart-3" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-foreground mb-2">Model Catalog</h3>
                  <p className="text-sm text-muted-foreground">
                    Auto-provision models from Hugging Face with checksum verify and local cache.
                  </p>
                </div>
              </div>
              <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                <span>Learn more</span>
                <ChevronRight className="size-3" />
              </div>
            </a>

            {/* Network Orchestration - Featured */}
            <a
              href="#cross-node-orchestration"
              className="group lg:col-span-1 md:col-span-2 lg:col-span-1 relative rounded-2xl border bg-card p-6 transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-150 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none overflow-hidden before:absolute before:inset-x-0 before:top-0 before:h-1.5 before:bg-gradient-to-r before:from-primary before:via-chart-3 before:to-amber-500"
              aria-label="Learn more about Network Orchestration"
            >
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-primary/10 p-3">
                  <Network className="size-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-foreground mb-2">Network Orchestration</h3>
                  <p className="text-sm text-muted-foreground">
                    Run jobs across gaming PCs, workstations, and Macs as one homelab cluster.
                  </p>
                </div>
              </div>
              <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                <span>Learn more</span>
                <ChevronRight className="size-3" />
              </div>
            </a>
          </div>
        </div>

        {/* Row 2: Developer Tools */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Badge variant="secondary">Developer Tools</Badge>
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {/* CLI & Web UI */}
            <a
              href="#cli-ui"
              className="group relative rounded-2xl border bg-card p-6 transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-200 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-muted-foreground"
              aria-label="Learn more about CLI & Web UI"
            >
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-muted/30 p-3">
                  <Terminal className="size-6 text-muted-foreground" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-foreground mb-2">CLI & Web UI</h3>
                  <p className="text-sm text-muted-foreground">
                    Automate with a fast CLI or manage visually in the web UI—your call.
                  </p>
                </div>
              </div>
              <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                <span>Learn more</span>
                <ChevronRight className="size-3" />
              </div>
            </a>

            {/* TypeScript SDK */}
            <a
              href="#sdk"
              className="group relative rounded-2xl border bg-card p-6 transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-300 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-primary"
              aria-label="Learn more about TypeScript SDK"
            >
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-primary/10 p-3">
                  <Code className="size-6 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-foreground mb-2">TypeScript SDK</h3>
                  <p className="text-sm text-muted-foreground">
                    Type-safe utilities for building agents; async/await with full IDE help.
                  </p>
                </div>
              </div>
              <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                <span>Learn more</span>
                <ChevronRight className="size-3" />
              </div>
            </a>

            {/* Security First */}
            <a
              href="#security-isolation"
              className="group relative rounded-2xl border bg-card p-6 transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-400 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-chart-2"
              aria-label="Learn more about Security First"
            >
              <div className="flex items-start gap-4">
                <div className="rounded-lg bg-chart-2/10 p-3">
                  <Shield className="size-6 text-chart-2" />
                </div>
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-foreground mb-2">Security First</h3>
                  <p className="text-sm text-muted-foreground">
                    Five Rust crates: auth, audit logs, input validation, secrets, and deadlines.
                  </p>
                </div>
              </div>
              <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                <span>Learn more</span>
                <ChevronRight className="size-3" />
              </div>
            </a>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
