'use client'

import { Anchor, DollarSign, Laptop, Shield } from 'lucide-react'
import { useEffect, useState } from 'react'
import { SectionContainer, FeatureCard, PledgeCallout, ArchitectureDiagram } from '@/components/molecules'
import { Badge } from '@/components/atoms/Badge/Badge'
import { Button } from '@/components/atoms/Button/Button'

export function SolutionSection() {
  const [prefersReduced, setPrefersReduced] = useState(false)

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)')
    setPrefersReduced(mediaQuery.matches)

    const handler = (e: MediaQueryListEvent) => setPrefersReduced(e.matches)
    mediaQuery.addEventListener('change', handler)
    return () => mediaQuery.removeEventListener('change', handler)
  }, [])

  const animClass = prefersReduced ? '' : 'animate-fade-in-up'

  return (
    <SectionContainer
      title={
        <>
          Your Hardware. Your Models. <span className="text-primary">Your Control.</span>
        </>
      }
      description="rbee orchestrates inference across every GPU in your home network—workstations, gaming rigs, and Macs—turning idle hardware into a private, OpenAI-compatible AI platform."
      bgVariant="secondary"
      maxWidth="7xl"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12 md:space-y-16">
        {/* Proof Badges */}
        <div className={`flex flex-wrap gap-2 justify-center ${animClass}`} style={{ animationDelay: '0ms' }}>
          <Badge variant="outline">OpenAI-compatible API</Badge>
          <Badge variant="outline">Runs on CUDA · Metal · CPU</Badge>
          <Badge variant="outline">Zero API fees (electricity only)</Badge>
          <Badge variant="outline">Code stays in your network</Badge>
        </div>

        {/* Architecture Diagram */}
        <div className={`max-w-4xl mx-auto ${animClass}`} style={{ animationDelay: '60ms' }}>
          <ArchitectureDiagram className="shadow-lg" />
        </div>

        {/* Key Benefits Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
          <div className={animClass} style={{ animationDelay: '120ms' }}>
            <FeatureCard
              icon={DollarSign}
              title="Zero Ongoing Costs"
              description="Pay only for electricity. No API bills, no per-token surprises."
              iconColor="chart-3"
              hover
              className="h-full hover:-translate-y-0.5 hover:shadow-md transition-transform"
            >
              <span className="text-xs text-muted-foreground">Typical: €0.08–€0.15/kWh</span>
            </FeatureCard>
          </div>

          <div className={animClass} style={{ animationDelay: '180ms' }}>
            <FeatureCard
              icon={Shield}
              title="Complete Privacy"
              description="Code and data never leave your network. Audit-ready by design."
              iconColor="chart-2"
              hover
              className="h-full hover:-translate-y-0.5 hover:shadow-md transition-transform"
            >
              <span className="text-xs text-muted-foreground">Audit-ready logs, EU-friendly</span>
            </FeatureCard>
          </div>

          <div className={animClass} style={{ animationDelay: '240ms' }}>
            <FeatureCard
              icon={Anchor}
              title="Locked to Your Rules"
              description="Models update only when you approve. No breaking changes."
              iconColor="primary"
              hover
              className="h-full hover:-translate-y-0.5 hover:shadow-md transition-transform"
            >
              <span className="text-xs text-muted-foreground">No forced updates</span>
            </FeatureCard>
          </div>

          <div className={animClass} style={{ animationDelay: '300ms' }}>
            <FeatureCard
              icon={Laptop}
              title="Use All Your Hardware"
              description="CUDA, Metal, and CPU orchestrated as one pool."
              iconColor="muted-foreground"
              hover
              className="h-full hover:-translate-y-0.5 hover:shadow-md transition-transform"
            >
              <span className="text-xs text-muted-foreground">Multi-node orchestration</span>
            </FeatureCard>
          </div>
        </div>

        {/* Privacy & Control Pledge */}
        <div className={animClass} style={{ animationDelay: '360ms' }}>
          <PledgeCallout />
        </div>

        {/* CTA Row */}
        <div className={`flex flex-col sm:flex-row gap-3 justify-center ${animClass}`} style={{ animationDelay: '420ms' }}>
          <Button size="lg" asChild>
            <a href="#quickstart">Run on my GPUs</a>
          </Button>
          <Button size="lg" variant="ghost" asChild>
            <a href="/docs/scheduler-policy">See scheduler policy</a>
          </Button>
        </div>
      </div>
    </SectionContainer>
  )
}
