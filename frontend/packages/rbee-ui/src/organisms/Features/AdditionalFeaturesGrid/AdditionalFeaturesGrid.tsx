import { Badge, Card, CardContent } from '@rbee/ui/atoms'
import { IconCardHeader, SectionContainer } from '@rbee/ui/molecules'
import { ChevronRight, Code, Database, Network, Shield, Terminal } from 'lucide-react'

export function AdditionalFeaturesGrid() {
  return (
    <SectionContainer
      title="Everything You Need for AI Infrastructure"
      bgVariant="background"
      eyebrow={<Badge variant="secondary">Capabilities overview</Badge>}
    >
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Row 1: Core Platform */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Badge variant="secondary">Core Platform</Badge>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {/* Cascading Shutdown */}
            <a
              href="#security-isolation"
              className="group block transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 focus-visible:outline-none"
              aria-label="Learn more about Cascading Shutdown"
            >
              <Card className="h-full before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-chart-2 before:rounded-t-xl relative overflow-hidden">
                <CardContent className="p-6">
                  <IconCardHeader
                    icon={Shield}
                    iconTone="chart-2"
                    iconSize="sm"
                    title="Cascading Shutdown"
                    subtitle="Ctrl+C tears down keeper → queen → hive → workers. No orphans, no VRAM leaks."
                    titleClassName="text-lg"
                    subtitleClassName="text-sm mt-2"
                    useCardHeader={false}
                  />
                  <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                    <span>Learn more</span>
                    <ChevronRight className="size-3" />
                  </div>
                </CardContent>
              </Card>
            </a>

            {/* Model Catalog */}
            <a
              href="#intelligent-model-management"
              className="group block transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-100 focus-visible:outline-none"
              aria-label="Learn more about Model Catalog"
            >
              <Card className="h-full before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-chart-3 before:rounded-t-xl relative overflow-hidden">
                <CardContent className="p-6">
                  <IconCardHeader
                    icon={Database}
                    iconTone="chart-3"
                    iconSize="sm"
                    title="Model Catalog"
                    subtitle="Auto-provision models from Hugging Face with checksum verify and local cache."
                    titleClassName="text-lg"
                    subtitleClassName="text-sm mt-2"
                    useCardHeader={false}
                  />
                  <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                    <span>Learn more</span>
                    <ChevronRight className="size-3" />
                  </div>
                </CardContent>
              </Card>
            </a>

            {/* Network Orchestration - Featured */}
            <a
              href="#cross-node-orchestration"
              className="group lg:col-span-1 md:col-span-2 lg:col-span-1 block transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-150 focus-visible:outline-none"
              aria-label="Learn more about Network Orchestration"
            >
              <Card className="h-full before:absolute before:inset-x-0 before:top-0 before:h-1.5 before:bg-gradient-to-r before:from-primary before:via-chart-3 before:to-amber-500 before:rounded-t-xl relative overflow-hidden">
                <CardContent className="p-6">
                  <IconCardHeader
                    icon={Network}
                    iconTone="primary"
                    iconSize="sm"
                    title="Network Orchestration"
                    subtitle="Run jobs across gaming PCs, workstations, and Macs as one homelab cluster."
                    titleClassName="text-lg"
                    subtitleClassName="text-sm mt-2"
                    useCardHeader={false}
                  />
                  <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                    <span>Learn more</span>
                    <ChevronRight className="size-3" />
                  </div>
                </CardContent>
              </Card>
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
              className="group block transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-200 focus-visible:outline-none"
              aria-label="Learn more about CLI & Web UI"
            >
              <Card className="h-full before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-muted-foreground before:rounded-t-xl relative overflow-hidden">
                <CardContent className="p-6">
                  <IconCardHeader
                    icon={Terminal}
                    iconTone="muted"
                    iconSize="sm"
                    title="CLI & Web UI"
                    subtitle="Automate with a fast CLI or manage visually in the web UI—your call."
                    titleClassName="text-lg"
                    subtitleClassName="text-sm mt-2"
                    useCardHeader={false}
                  />
                  <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                    <span>Learn more</span>
                    <ChevronRight className="size-3" />
                  </div>
                </CardContent>
              </Card>
            </a>

            {/* TypeScript SDK */}
            <a
              href="#sdk"
              className="group block transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-300 focus-visible:outline-none"
              aria-label="Learn more about TypeScript SDK"
            >
              <Card className="h-full before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-primary before:rounded-t-xl relative overflow-hidden">
                <CardContent className="p-6">
                  <IconCardHeader
                    icon={Code}
                    iconTone="primary"
                    iconSize="sm"
                    title="TypeScript SDK"
                    subtitle="Type-safe utilities for building agents; async/await with full IDE help."
                    titleClassName="text-lg"
                    subtitleClassName="text-sm mt-2"
                    useCardHeader={false}
                  />
                  <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                    <span>Learn more</span>
                    <ChevronRight className="size-3" />
                  </div>
                </CardContent>
              </Card>
            </a>

            {/* Security First */}
            <a
              href="#security-isolation"
              className="group block transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 delay-400 focus-visible:outline-none"
              aria-label="Learn more about Security First"
            >
              <Card className="h-full before:absolute before:inset-x-0 before:top-0 before:h-[2px] before:bg-chart-2 before:rounded-t-xl relative overflow-hidden">
                <CardContent className="p-6">
                  <IconCardHeader
                    icon={Shield}
                    iconTone="chart-2"
                    iconSize="sm"
                    title="Security First"
                    subtitle="Six Rust crates: auth, audit logs, input validation, secrets, JWT guardian, and deadlines."
                    titleClassName="text-lg"
                    subtitleClassName="text-sm mt-2"
                    useCardHeader={false}
                  />
                  <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                    <span>Learn more</span>
                    <ChevronRight className="size-3" />
                  </div>
                </CardContent>
              </Card>
            </a>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
