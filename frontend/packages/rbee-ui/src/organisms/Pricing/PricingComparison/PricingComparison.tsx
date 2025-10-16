import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import type { Provider, Row, RowGroup } from '@rbee/ui/molecules'
import { MatrixTable } from '@rbee/ui/molecules'
import { Check, X } from 'lucide-react'

// Created by: TEAM-086

interface PricingComparisonProps {
  lastUpdated?: string
}

// Define columns (providers) for the comparison table
const columns: Provider[] = [
  {
    key: 'h',
    label: 'Home/Lab',
    subtitle: 'Solo / Homelab',
  },
  {
    key: 't',
    label: 'Team',
    subtitle: 'Small teams',
    badge: 'Best for most teams',
    accent: true,
  },
  {
    key: 'e',
    label: 'Enterprise',
    subtitle: 'Security & SLA',
  },
]

// Define feature groups
const groups: RowGroup[] = [
  { id: 'core', label: 'Core Platform' },
  { id: 'productivity', label: 'Productivity' },
  { id: 'support', label: 'Support & Services' },
]

// Define features (rows) with values for each column
const rows: Row[] = [
  // Core Platform
  {
    feature: 'Number of GPUs',
    group: 'core',
    values: { h: 'Unlimited', t: 'Unlimited', e: 'Unlimited' },
  },
  {
    feature: 'OpenAI-compatible API',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  {
    feature: 'Multi-GPU orchestration (one or many nodes)',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  {
    feature: 'Programmable routing (Rhai scheduler)',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  {
    feature: 'CLI access',
    group: 'core',
    values: { h: true, t: true, e: true },
  },
  // Productivity
  {
    feature: 'Web UI (manage nodes, models, jobs)',
    group: 'productivity',
    values: { h: false, t: true, e: true },
  },
  {
    feature: 'Team collaboration',
    group: 'productivity',
    values: { h: false, t: true, e: true },
  },
  // Support & Services
  {
    feature: 'Support',
    group: 'support',
    values: {
      h: 'Community',
      t: 'Priority email (business hours)',
      e: 'Dedicated (SLA-backed)',
    },
  },
  {
    feature: 'SLA',
    group: 'support',
    values: { h: false, t: false, e: true },
    note: 'Response and uptime commitments (Enterprise only).',
  },
  {
    feature: 'White-label',
    group: 'support',
    values: { h: false, t: false, e: true },
  },
  {
    feature: 'Professional services',
    group: 'support',
    values: { h: false, t: false, e: true },
  },
]

export function PricingComparison({ lastUpdated }: PricingComparisonProps) {
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

        {/* Comparison Table */}
        <Card className="rounded-xl border border-border bg-card shadow-sm overflow-hidden p-0">
          <div className="overflow-x-auto">
            <MatrixTable
              columns={columns}
              rows={rows}
              groups={groups}
              caption="Feature availability comparison across Home/Lab, Team, and Enterprise plans."
            />
          </div>
        </Card>

        {/* CTA Strip */}
        <div className="mt-6 flex flex-wrap items-center gap-3 justify-between rounded-xl border border-border p-4 bg-secondary animate-in fade-in slide-in-from-bottom-2 duration-500">
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
