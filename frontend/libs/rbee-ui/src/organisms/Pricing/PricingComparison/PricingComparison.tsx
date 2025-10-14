import { Check, X } from 'lucide-react'
import { Badge } from '@rbee/ui/atoms/Badge'
import { Card } from '@rbee/ui/atoms/Card'
import { Separator } from '@rbee/ui/atoms/Separator'
import { Tooltip, TooltipContent, TooltipTrigger } from '@rbee/ui/atoms/Tooltip'
import { Button } from '@rbee/ui/atoms/Button'
import { cn } from '@rbee/ui/utils'

// Created by: TEAM-086

interface PricingComparisonProps {
  lastUpdated?: string
}

interface Feature {
  key: string
  label: string
  group: 'core' | 'productivity' | 'support'
  h: boolean | string
  t: boolean | string
  e: boolean | string
  note?: string
}

const features: Feature[] = [
  // Core Platform
  { key: 'gpus', label: 'Number of GPUs', group: 'core', h: 'Unlimited', t: 'Unlimited', e: 'Unlimited' },
  { key: 'api', label: 'OpenAI-compatible API', group: 'core', h: true, t: true, e: true },
  {
    key: 'orchestration',
    label: 'Multi-GPU orchestration (one or many nodes)',
    group: 'core',
    h: true,
    t: true,
    e: true,
  },
  { key: 'scheduler', label: 'Programmable routing (Rhai scheduler)', group: 'core', h: true, t: true, e: true },
  { key: 'cli', label: 'CLI access', group: 'core', h: true, t: true, e: true },
  // Productivity
  { key: 'webui', label: 'Web UI (manage nodes, models, jobs)', group: 'productivity', h: false, t: true, e: true },
  { key: 'collab', label: 'Team collaboration', group: 'productivity', h: false, t: true, e: true },
  // Support & Services
  {
    key: 'support',
    label: 'Support',
    group: 'support',
    h: 'Community',
    t: 'Priority email (business hours)',
    e: 'Dedicated (SLA-backed)',
  },
  { key: 'sla', label: 'SLA', group: 'support', h: false, t: false, e: true, note: 'Response and uptime commitments (Enterprise only).' },
  { key: 'whitelabel', label: 'White-label', group: 'support', h: false, t: false, e: true },
  { key: 'services', label: 'Professional services', group: 'support', h: false, t: false, e: true },
]

const groupLabels = {
  core: 'Core Platform',
  productivity: 'Productivity',
  support: 'Support & Services',
}

function PlanCell({ included, emphasis = false }: { included: boolean | string; emphasis?: boolean }) {
  if (typeof included === 'string') {
    return (
      <td className={cn('px-4 py-2.5 text-center text-sm', emphasis && 'bg-primary/5')}>
        <span className="text-foreground">{included}</span>
      </td>
    )
  }

  const Icon = included ? Check : X

  return (
    <td className={cn('px-4 py-2.5 text-center', emphasis && 'bg-primary/5')}>
      <Icon className={cn('h-5 w-5 mx-auto', included ? 'text-chart-3' : 'text-muted-foreground/40')} />
      <span className="sr-only">{included ? 'Included' : 'Not available'}</span>
    </td>
  )
}

function FeatureRow({ feature }: { feature: Feature }) {
  const label = feature.note ? (
    <Tooltip>
      <TooltipTrigger asChild>
        <button className="text-left underline decoration-dotted underline-offset-4 hover:text-foreground transition-colors">
          {feature.label}
        </button>
      </TooltipTrigger>
      <TooltipContent>{feature.note}</TooltipContent>
    </Tooltip>
  ) : (
    feature.label
  )

  return (
    <tr className="odd:bg-muted/30 border-b border-border hover:bg-accent/40 transition-colors duration-200">
      <td className="px-4 py-2.5 text-muted-foreground sticky left-0 z-10 bg-card text-sm">{label}</td>
      <PlanCell included={feature.h} />
      <PlanCell included={feature.t} emphasis />
      <PlanCell included={feature.e} />
    </tr>
  )
}

function RowGroupLabel({ label }: { label: string }) {
  return (
    <tr className="bg-secondary/50">
      <th scope="rowgroup" colSpan={4} className="text-left px-4 py-2 text-sm font-semibold text-foreground">
        {label}
      </th>
    </tr>
  )
}

function MobileFeatureCard({ feature }: { feature: Feature }) {
  const renderBadge = (included: boolean | string, plan: string) => {
    if (typeof included === 'string') {
      return (
        <Badge variant="outline" className="rounded-full px-2.5 py-1 text-xs border bg-secondary text-secondary-foreground">
          {plan}: {included}
        </Badge>
      )
    }

    const Icon = included ? Check : X
    return (
      <Badge
        variant="outline"
        className={cn(
          'inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs border bg-secondary text-secondary-foreground',
        )}
      >
        <Icon className={cn('h-3 w-3', included ? 'text-chart-3' : 'text-muted-foreground/40')} />
        {plan}
      </Badge>
    )
  }

  return (
    <Card className="p-4">
      <div className="font-medium text-sm mb-3">{feature.label}</div>
      <div className="flex flex-wrap gap-2">
        {renderBadge(feature.h, 'Home/Lab')}
        {renderBadge(feature.t, 'Team')}
        {renderBadge(feature.e, 'Enterprise')}
      </div>
    </Card>
  )
}

export function PricingComparison({ lastUpdated }: PricingComparisonProps) {
  const groupedFeatures = {
    core: features.filter((f) => f.group === 'core'),
    productivity: features.filter((f) => f.group === 'productivity'),
    support: features.filter((f) => f.group === 'support'),
  }

  return (
    <section className="py-16 bg-secondary">
      <div className="max-w-6xl mx-auto px-6 sm:px-8">
        {/* Header */}
        <div className="grid grid-cols-1 md:grid-cols-2 items-start gap-6 mb-6 animate-in fade-in slide-in-from-bottom-1 duration-500 ease-out">
          {/* Left: Title + Value Prop */}
          <div>
            <h2 className="text-3xl font-bold tracking-tight mb-2">Detailed Feature Comparison</h2>
            <p className="text-sm text-muted-foreground">What changes across Home/Lab, Team, and Enterprise.</p>
          </div>

          {/* Right: Legend + Decisions Panel */}
          <div className="space-y-4">
            {/* Legend */}
            <div className="text-sm text-muted-foreground">
              <div className="flex items-center gap-3 mb-1">
                <span className="inline-flex items-center gap-1">
                  <Check className="h-4 w-4 text-chart-3" />
                  Included
                </span>
                <span>•</span>
                <span className="inline-flex items-center gap-1">
                  <X className="h-4 w-4 text-muted-foreground/40" />
                  Not available
                </span>
              </div>
              <div className="text-xs">Last updated: {lastUpdated || 'This month'}</div>
            </div>

            {/* Decisions Panel */}
            <Card className="p-4 text-sm text-muted-foreground leading-relaxed space-y-2">
              <div className="font-semibold text-foreground mb-2">Key Differences</div>
              <ul className="space-y-1.5 list-disc list-inside">
                <li>Team adds Web UI + collaboration</li>
                <li>Enterprise adds SLA + white-label + services</li>
                <li>All plans support unlimited GPUs</li>
              </ul>
            </Card>
          </div>
        </div>

        {/* Desktop Table */}
        <Card className="rounded-xl border border-border bg-card shadow-sm overflow-hidden hidden md:block p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <caption className="sr-only">
                Feature availability comparison across Home/Lab, Team, and Enterprise plans.
              </caption>
              <thead>
                <tr className="border-b border-border">
                  <th scope="col" className="text-left px-4 py-3 font-semibold text-foreground sticky left-0 z-10 bg-card">
                    Feature
                  </th>
                  <th scope="col" className="px-4 py-3 text-center">
                    <div className="font-semibold">Home/Lab</div>
                    <div className="text-xs text-muted-foreground font-normal">Solo / Homelab</div>
                  </th>
                  <th scope="col" className="px-4 py-3 text-center bg-primary/5 ring-1 ring-primary/10">
                    <div className="font-semibold">Team</div>
                    <div className="text-xs text-muted-foreground font-normal">Small teams</div>
                    <Badge variant="outline" className="mt-1 rounded-full text-xs animate-in fade-in">
                      Best for most teams
                    </Badge>
                  </th>
                  <th scope="col" className="px-4 py-3 text-center">
                    <div className="font-semibold">Enterprise</div>
                    <div className="text-xs text-muted-foreground font-normal">Security & SLA</div>
                  </th>
                </tr>
              </thead>
              <tbody>
                <RowGroupLabel label={groupLabels.core} />
                {groupedFeatures.core.map((feature) => (
                  <FeatureRow key={feature.key} feature={feature} />
                ))}

                <tr>
                  <td colSpan={4}>
                    <Separator className="my-1 opacity-50" />
                  </td>
                </tr>

                <RowGroupLabel label={groupLabels.productivity} />
                {groupedFeatures.productivity.map((feature) => (
                  <FeatureRow key={feature.key} feature={feature} />
                ))}

                <tr>
                  <td colSpan={4}>
                    <Separator className="my-1 opacity-50" />
                  </td>
                </tr>

                <RowGroupLabel label={groupLabels.support} />
                {groupedFeatures.support.map((feature) => (
                  <FeatureRow key={feature.key} feature={feature} />
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Mobile Card List */}
        <div className="md:hidden space-y-3">
          {features.map((feature) => (
            <MobileFeatureCard key={feature.key} feature={feature} />
          ))}
        </div>

        {/* CTA Strip */}
        <div className="mt-6 flex flex-wrap items-center gap-3 justify-between rounded-xl border p-4 bg-secondary animate-in fade-in slide-in-from-bottom-2 duration-500">
          <div className="text-sm font-medium">Ready to get started?</div>
          <div className="flex gap-3">
            <Button size="default">Start with Team</Button>
            <Button variant="outline" size="default" asChild>
              <a href="/contact">Talk to Sales</a>
            </Button>
          </div>
        </div>
      </div>
    </section>
  )
}
