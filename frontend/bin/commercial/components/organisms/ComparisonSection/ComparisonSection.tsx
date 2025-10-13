import { Check, X } from 'lucide-react'
import Link from 'next/link'
import { SectionContainer } from '@/components/molecules'
import { Button } from '@/components/atoms/Button/Button'

const features = [
  {
    name: 'Total Cost',
    rbee: { value: '$0', note: 'runs on your hardware', badge: 'Lowest' },
    openai: '$20–100/mo per dev',
    ollama: '$0',
    runpod: '$0.50–2/hr',
  },
  {
    name: 'Privacy / Data Residency',
    rbee: { icon: 'check', note: 'Complete' },
    openai: { icon: 'x', note: 'Limited' },
    ollama: { icon: 'check', note: 'Complete' },
    runpod: { icon: 'x', note: 'Limited' },
  },
  {
    name: 'Multi-GPU Utilization',
    rbee: { icon: 'check', note: 'Orchestrated', tooltip: 'Unified pool across CUDA, Metal, CPU' },
    openai: 'N/A',
    ollama: 'Limited',
    runpod: { icon: 'check' },
  },
  {
    name: 'OpenAI-Compatible API',
    rbee: { icon: 'check' },
    openai: { icon: 'check' },
    ollama: { value: 'Partial', tooltip: 'Some endpoints missing' },
    runpod: { icon: 'x' },
  },
  {
    name: 'Custom Routing Policies',
    rbee: { icon: 'check', note: 'Rhai-based policies', tooltip: 'Script routing by model, region, cost' },
    openai: { icon: 'x' },
    ollama: { icon: 'x' },
    runpod: { icon: 'x' },
  },
  {
    name: 'Rate Limits / Quotas',
    rbee: { value: 'None', positive: true },
    openai: { value: 'Yes', negative: true },
    ollama: { value: 'None', positive: true },
    runpod: { value: 'Yes', negative: true },
  },
]

function CellContent({ data }: { data: any }) {
  if (typeof data === 'string') {
    return <div className="text-sm text-muted-foreground">{data}</div>
  }
  if (data.icon === 'check') {
    return (
      <div className="flex flex-col items-center gap-1">
        <Check className="h-5 w-5 text-chart-3" aria-hidden="true" />
        <span className="sr-only">Available</span>
        {data.note && <div className="text-xs text-muted-foreground">{data.note}</div>}
      </div>
    )
  }
  if (data.icon === 'x') {
    return (
      <div className="flex flex-col items-center gap-1">
        <X className="h-5 w-5 text-destructive" aria-hidden="true" />
        <span className="sr-only">Not available</span>
        {data.note && <div className="text-xs text-muted-foreground">{data.note}</div>}
      </div>
    )
  }
  if (data.value) {
    const colorClass = data.positive ? 'text-chart-3' : data.negative ? 'text-destructive' : 'text-card-foreground'
    return (
      <div className="flex flex-col items-center gap-1">
        <div className={`text-sm font-medium ${colorClass}`}>{data.value}</div>
        {data.note && <div className="text-xs text-muted-foreground">{data.note}</div>}
        {data.badge && (
          <span className="inline-flex items-center bg-chart-3/10 text-chart-3 text-[11px] px-2 py-0.5 rounded-full font-medium">
            {data.badge}
          </span>
        )}
      </div>
    )
  }
  return null
}

export function ComparisonSection() {
  return (
    <SectionContainer
      title="Why Developers Choose rbee"
      subtitle="Local-first AI that's faster, private, and costs $0 on your hardware."
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

        {/* Desktop table */}
        <div className="hidden md:block max-w-6xl mx-auto overflow-x-auto rounded-xl ring-1 ring-border/60 bg-card">
          <table className="w-full">
            <caption className="sr-only">Comparison of rbee vs cloud/API options</caption>
            <colgroup>
              <col className="w-40 sm:w-56" />
              <col className="bg-primary/5" />
              <col />
              <col />
              <col />
            </colgroup>
            <thead>
              <tr className="border-b border-border sticky top-0 z-10 backdrop-blur supports-[backdrop-filter]:bg-card/80 bg-card">
                <th scope="col" className="text-left p-4 font-bold text-card-foreground">
                  Feature
                </th>
                <th scope="col" className="p-4 font-bold text-primary bg-primary/5 relative">
                  <div>rbee</div>
                  <div className="absolute bottom-0 left-1/2 -translate-x-1/2 h-0.5 w-16 bg-primary rounded-full" />
                </th>
                <th scope="col" className="p-4 font-medium text-muted-foreground">
                  OpenAI & Anthropic
                </th>
                <th scope="col" className="p-4 font-medium text-muted-foreground">
                  Ollama
                </th>
                <th scope="col" className="p-4 font-medium text-muted-foreground">
                  Runpod & Vast.ai
                </th>
              </tr>
            </thead>
            <tbody>
              {features.map((feature, index) => (
                <tr
                  key={feature.name}
                  className="border-b border-border last:border-b-0 odd:bg-muted/20 even:bg-card hover:bg-muted/30 transition-colors animate-in slide-in-from-bottom-1 duration-400"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <th scope="row" className="text-left p-4 font-medium text-card-foreground">
                    {feature.name}
                  </th>
                  <td className="p-4 text-center bg-primary/5">
                    <CellContent data={feature.rbee} />
                  </td>
                  <td className="p-4 text-center">
                    <CellContent data={feature.openai} />
                  </td>
                  <td className="p-4 text-center">
                    <CellContent data={feature.ollama} />
                  </td>
                  <td className="p-4 text-center">
                    <CellContent data={feature.runpod} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Mobile stacked view */}
        <div className="md:hidden space-y-4">
          {features.map((feature, index) => (
            <div
              key={feature.name}
              className="bg-card border border-border rounded-xl p-4 space-y-3 animate-in slide-in-from-bottom-2 duration-400"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <h3 className="font-semibold text-card-foreground text-sm">{feature.name}</h3>
              <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-sm">
                <div className="text-muted-foreground font-medium">rbee</div>
                <div className="flex items-center justify-end">
                  <CellContent data={feature.rbee} />
                </div>

                <div className="text-muted-foreground font-medium">OpenAI & Anthropic</div>
                <div className="flex items-center justify-end">
                  <CellContent data={feature.openai} />
                </div>

                <div className="text-muted-foreground font-medium">Ollama</div>
                <div className="flex items-center justify-end">
                  <CellContent data={feature.ollama} />
                </div>

                <div className="text-muted-foreground font-medium">Runpod & Vast.ai</div>
                <div className="flex items-center justify-end">
                  <CellContent data={feature.runpod} />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Footer CTA */}
        <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-center items-center">
          <p className="text-sm text-muted-foreground text-center sm:text-left">
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
