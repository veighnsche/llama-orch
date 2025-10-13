import { type ReactNode } from 'react'
import { cn } from '@/lib/utils'
import { BeeArchitecture, type BeeTopology } from '@/components/molecules'

export type Benefit = {
  icon: ReactNode
  title: string
  body: string
}

export interface HomeSolutionSectionProps {
  title: string
  subtitle?: string
  benefits: Benefit[]
  topology: BeeTopology
  id?: string
  className?: string
}

/**
 * HomeSolutionSection - Specialized solution section for the home page
 * with BeeArchitecture diagram. For conversion-focused solution sections
 * (providers, developers, enterprise), use SolutionSection instead.
 */
export function HomeSolutionSection({ title, subtitle, benefits, topology, id, className }: HomeSolutionSectionProps) {
  return (
    <section id={id} className={cn('border-b border-border py-24', className)}>
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        {/* Header */}
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">{title}</h2>
          {subtitle && <p className="text-balance text-lg leading-relaxed text-muted-foreground">{subtitle}</p>}
        </div>

        {/* Benefits Grid */}
        <div className="mx-auto mt-16 grid max-w-5xl gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {benefits.map((benefit, idx) => (
            <div
              key={idx}
              className="group rounded-lg border border-border bg-card p-6 transition-all hover:border-primary/50 hover:bg-card/80"
            >
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                {benefit.icon}
              </div>
              <h3 className="mb-2 text-lg font-semibold text-card-foreground">{benefit.title}</h3>
              <p className="text-balance text-sm leading-relaxed text-muted-foreground">{benefit.body}</p>
            </div>
          ))}
        </div>

        {/* Architecture Diagram */}
        <BeeArchitecture topology={topology} />
      </div>
    </section>
  )
}
