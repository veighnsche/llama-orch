'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { ComplianceShield, DevGrid, GpuMarket } from '@rbee/ui/icons'
import { AudienceCard, SectionContainer } from '@rbee/ui/molecules'
import { ChevronRight, Code2, Server, Shield } from 'lucide-react'
import Link from 'next/link'

export function AudienceSelector() {
  return (
    <SectionContainer
      eyebrow="Choose your path"
      title="Where should you start?"
      description="rbee adapts to how you work—build on your own GPUs, monetize idle capacity, or deploy compliant AI at scale."
      bgVariant="subtle"
      paddingY="2xl"
      maxWidth="7xl"
      align="center"
    >
      {/* Radial gradient backplate */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-[600px] opacity-40"
        style={{
          background: 'radial-gradient(ellipse 80% 50% at 50% 0%, hsl(var(--primary) / 0.05), transparent)',
        }}
        aria-hidden="true"
      />

      {/* Grid with responsive 2→3 column layout and equal heights */}
      <div
        className="mx-auto grid max-w-6xl grid-cols-1 content-start gap-6 sm:grid-cols-2 xl:grid-cols-3 xl:gap-8"
        aria-label="Audience options: Developers, GPU Owners, Enterprise"
      >
        {/* Developers Card */}
        <div className="flex h-full">
          <AudienceCard
            icon={Code2}
            category="For Developers"
            title="Build on Your Hardware"
            description="Power Zed, Cursor, and your own agents on YOUR GPUs. OpenAI-compatible—drop-in, zero API fees."
            features={[
              'Zero API costs, unlimited usage',
              'Your code stays on your network',
              'Agentic API + TypeScript utils',
            ]}
            href="/developers"
            ctaText="Explore Developer Path"
            color="chart-2"
            imageSlot={<DevGrid size={56} aria-hidden />}
            badgeSlot={
              <Badge variant="outline" className="border-chart-2/30 bg-chart-2/5 text-chart-2">
                Homelab-ready
              </Badge>
            }
            decisionLabel="Code with AI locally"
          />
        </div>

        {/* GPU Owners Card */}
        <div className="flex h-full">
          <AudienceCard
            icon={Server}
            category="For GPU Owners"
            title="Monetize Your Hardware"
            description="Join the rbee marketplace and earn from gaming rigs to server farms—set price, stay in control."
            features={['Set pricing & availability', 'Audit trails and payouts', 'Passive income from idle GPUs']}
            href="/gpu-providers"
            ctaText="Become a Provider"
            color="chart-3"
            imageSlot={<GpuMarket size={56} aria-hidden />}
            decisionLabel="Earn from idle GPUs"
          />
        </div>

        {/* Enterprise Card */}
        <div className="flex h-full">
          <AudienceCard
            icon={Shield}
            category="For Enterprise"
            title="Compliance & Security"
            description="EU-native compliance, audit trails, and zero-trust architecture—from day one."
            features={['GDPR with 7-year retention', 'SOC2 & ISO 27001 aligned', 'Private cloud or on-prem']}
            href="/enterprise"
            ctaText="Enterprise Solutions"
            color="primary"
            imageSlot={<ComplianceShield size={56} aria-hidden />}
            decisionLabel="Deploy with compliance"
          />
        </div>
      </div>

      {/* Bottom helper links */}
      <div className="mx-auto mt-12 text-center">
        <Link
          href="#compare"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40"
        >
          Not sure? Compare paths
          <ChevronRight className="h-3.5 w-3.5" />
        </Link>
        <span className="mx-3 text-muted-foreground/50">·</span>
        <Link
          href="#contact"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40"
        >
          Talk to us
          <ChevronRight className="h-3.5 w-3.5" />
        </Link>
      </div>
    </SectionContainer>
  )
}
