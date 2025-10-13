import { Building, Home, Laptop, Users } from 'lucide-react'
import Link from 'next/link'
import { SectionContainer, FeatureCard } from '@/components/molecules'
import { Badge } from '@/components/atoms/Badge/Badge'
import { Button } from '@/components/atoms/Button/Button'

export function UseCasesSection() {
  return (
    <SectionContainer
      title="Built for Those Who Value Independence"
      subtitle="Run serious AI on your own hardware. Keep costs at zero, keep control at 100%."
      bgVariant="secondary"
    >
      <div className="max-w-6xl mx-auto space-y-8 animate-in fade-in-50 duration-500">
        {/* Audience key strip */}
        <div className="flex flex-wrap gap-2 text-sm text-muted-foreground justify-center">
          <Badge variant="outline" asChild>
            <a href="#usecase-solo" className="hover:bg-accent transition-colors">
              Solo
            </a>
          </Badge>
          <Badge variant="outline" asChild>
            <a href="#usecase-team" className="hover:bg-accent transition-colors">
              Small Team
            </a>
          </Badge>
          <Badge variant="outline" asChild>
            <a href="#usecase-homelab" className="hover:bg-accent transition-colors">
              Homelab
            </a>
          </Badge>
          <Badge variant="outline" asChild>
            <a href="#usecase-enterprise" className="hover:bg-accent transition-colors">
              Enterprise
            </a>
          </Badge>
        </div>

        {/* Cards grid */}
        <div className="grid grid-cols-12 gap-6 lg:gap-8">
          <div className="col-span-12 md:col-span-6 animate-in slide-in-from-bottom-2 duration-500">
            <FeatureCard
              id="usecase-solo"
              icon={Laptop}
              title="The Solo Developer"
              description="Shipping a SaaS with AI help but allergic to lock-in."
              iconColor="chart-2"
              size="lg"
              className="bg-secondary/60 hover:bg-secondary transition-colors border-border/60 h-full"
              stat={{ label: 'Monthly AI cost', value: '$0' }}
              bullets={[
                'Run rbee on your gaming PC + spare workstation.',
                'Llama 70B for coding, SD for assets—local & fast.',
                'No rate limits. Your tools, your rules.',
              ]}
            />
          </div>

          <div className="col-span-12 md:col-span-6 animate-in slide-in-from-bottom-2 duration-500 delay-100">
            <FeatureCard
              id="usecase-team"
              icon={Users}
              title="The Small Team"
              description="5-person startup burning $500/mo on APIs."
              iconColor="primary"
              size="lg"
              className="bg-secondary/60 hover:bg-secondary transition-colors border-border/60 h-full"
              stat={{ label: 'Savings/yr', value: '$6,000+' }}
              bullets={[
                'Pool 3 workstations + 2 Macs into one rbee cluster.',
                'Shared models, faster inference, fewer blockers.',
                'GDPR-friendly by design.',
              ]}
            />
          </div>

          <div className="col-span-12 md:col-span-6 animate-in slide-in-from-bottom-2 duration-500 delay-200">
            <FeatureCard
              id="usecase-homelab"
              icon={Home}
              title="The Homelab Enthusiast"
              description="Four GPUs gathering dust."
              iconColor="chart-3"
              size="lg"
              className="bg-secondary/60 hover:bg-secondary transition-colors border-border/60 h-full"
              stat={{ label: 'Idle GPUs', value: '→ Productive' }}
              bullets={[
                'Spread workers across your LAN in minutes.',
                'Build agents: coder, doc generator, code reviewer.',
                'Auto-download models, clean shutdowns, no mess.',
              ]}
            />
          </div>

          <div className="col-span-12 md:col-span-6 animate-in slide-in-from-bottom-2 duration-500 delay-300">
            <FeatureCard
              id="usecase-enterprise"
              icon={Building}
              title="The Enterprise"
              description="50-dev org. Code can't leave the premises."
              iconColor="chart-4"
              size="lg"
              className="bg-secondary/60 hover:bg-secondary transition-colors border-border/60 h-full"
              stat={{ label: 'Compliance', value: 'EU-only' }}
              bullets={[
                'On-prem rbee with audit trails and policy routing.',
                'Rhai-based rules for data residency & access.',
                'Zero external dependencies.',
              ]}
            />
          </div>
        </div>

        {/* CTA and reassurance footer */}
        <div className="mt-8 flex flex-col sm:flex-row items-start sm:items-center gap-4 justify-center">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <Button asChild size="lg">
              <Link href="/docs/quickstart">See Quickstart</Link>
            </Button>
            <Button asChild variant="ghost" size="lg">
              <Link href="/docs/architecture">
                Architecture →
              </Link>
            </Button>
          </div>
        </div>

        {/* Micro badges */}
        <div className="flex flex-wrap gap-2 justify-center">
          <Badge variant="secondary" className="text-xs">
            OpenAI-compatible
          </Badge>
          <Badge variant="secondary" className="text-xs">
            Multi-backend (CUDA/Metal/CPU)
          </Badge>
          <Badge variant="secondary" className="text-xs">
            Audit-ready
          </Badge>
        </div>

        <p className="text-center text-sm text-muted-foreground">
          Works with Zed, Cursor, and any OpenAI-compatible tool.
        </p>
      </div>
    </SectionContainer>
  )
}
