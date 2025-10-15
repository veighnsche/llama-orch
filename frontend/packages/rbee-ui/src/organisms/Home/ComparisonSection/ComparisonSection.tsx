import { Button } from '@rbee/ui/atoms/Button'
import { SectionContainer } from '@rbee/ui/molecules'
import type { Provider, Row } from '@rbee/ui/molecules/Tables/MatrixTable'
import { MatrixTable } from '@rbee/ui/molecules/Tables/MatrixTable'
import { Check, X } from 'lucide-react'
import Link from 'next/link'

const columns: Provider[] = [
  { key: 'rbee', label: 'rbee', accent: true },
  { key: 'openai', label: 'OpenAI & Anthropic' },
  { key: 'ollama', label: 'Ollama' },
  { key: 'runpod', label: 'Runpod & Vast.ai' },
]

const rows: Row[] = [
  {
    feature: 'Total Cost',
    values: {
      rbee: '$0 (runs on your hardware)',
      openai: '$20–100/mo per dev',
      ollama: '$0',
      runpod: '$0.50–2/hr',
    },
  },
  {
    feature: 'Privacy / Data Residency',
    values: {
      rbee: true,
      openai: false,
      ollama: true,
      runpod: false,
    },
    note: 'Complete data control vs. limited',
  },
  {
    feature: 'Multi-GPU Utilization',
    values: {
      rbee: true,
      openai: 'N/A',
      ollama: 'Limited',
      runpod: true,
    },
  },
  {
    feature: 'OpenAI-Compatible API',
    values: {
      rbee: true,
      openai: true,
      ollama: 'Partial',
      runpod: false,
    },
  },
  {
    feature: 'Custom Routing Policies',
    values: {
      rbee: true,
      openai: false,
      ollama: false,
      runpod: false,
    },
  },
  {
    feature: 'Rate Limits / Quotas',
    values: {
      rbee: 'None',
      openai: 'Yes',
      ollama: 'None',
      runpod: 'Yes',
    },
  },
]

export function ComparisonSection() {
  return (
    <SectionContainer
      title="Why Developers Choose rbee"
      description="Local-first AI that's faster, private, and costs $0 on your hardware."
      bgVariant="secondary"
    >
      <div className="max-w-6xl mx-auto space-y-6 animate-in fade-in-50 duration-500">
        {/* Legend */}
        <div className="text-xs text-muted-foreground flex flex-wrap gap-4 justify-center">
          <span className="flex items-center gap-1.5">
            <Check className="h-3.5 w-3.5 text-chart-3" aria-hidden="true" />
            Available
          </span>
          <span className="flex items-center gap-1.5">
            <X className="h-3.5 w-3.5 text-destructive" aria-hidden="true" />
            Not available
          </span>
          <span>"Partial" = limited coverage</span>
        </div>

        {/* Comparison table */}
        <div className="rounded-xl ring-1 ring-border/60 bg-card overflow-hidden">
          <MatrixTable columns={columns} rows={rows} />
        </div>

        {/* Footer CTA */}
        <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center items-center">
          <p className="text-sm text-muted-foreground text-center sm:text-left font-sans">
            Bring your own GPUs, keep your data in-house.
          </p>
          <div className="flex gap-3">
            <Button asChild size="default">
              <Link href="/docs/quickstart">See Quickstart</Link>
            </Button>
            <Button asChild variant="ghost" size="default">
              <Link href="/docs/architecture">Architecture</Link>
            </Button>
          </div>
        </div>
      </div>
    </SectionContainer>
  )
}
