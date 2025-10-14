import { type ReactNode } from 'react'
import Image from 'next/image'
import { cn } from '@rbee/ui/utils'
import { CodeSnippet } from '@rbee/ui/atoms'

export type Step = {
  icon: ReactNode
  step: string // "Step 1"
  title: string
  body: string
  checklist?: string[] // optional
  snippet?: string // optional CodeSnippet content
  successNote?: string // optional green badge
  stats?: { label: string; value: string }[] // optional small stats grid
}

export type StepsSectionProps = {
  kicker?: string
  title: string
  subtitle?: string
  steps: Step[]
  avgTime?: string // e.g., "12 minutes"
  diagramSrc?: string // optional image
  id?: string
  className?: string
}

export function StepsSection({
  kicker,
  title,
  subtitle,
  steps,
  avgTime,
  diagramSrc,
  id,
  className,
}: StepsSectionProps) {
  return (
    <section
      id={id}
      className={cn(
        'border-b border-border bg-gradient-to-b from-background via-primary/5 to-card px-6 py-20 lg:py-28',
        className,
      )}
    >
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-12 text-center animate-in fade-in slide-in-from-bottom-2 duration-500 motion-reduce:animate-none">
          {kicker && <p className="mb-2 text-sm font-medium text-primary/80">{kicker}</p>}
          <h2 className="text-balance text-4xl font-extrabold tracking-tight lg:text-5xl">{title}</h2>
          {subtitle && (
            <p className="mx-auto mt-4 max-w-2xl text-lg leading-snug text-muted-foreground lg:text-xl">{subtitle}</p>
          )}
        </div>

        {/* Optional Diagram */}
        {diagramSrc && (
          <Image
            src={diagramSrc}
            width={1080}
            height={360}
            className="mx-auto mb-8 hidden rounded-xl border border-border/70 shadow-sm lg:block"
            alt="Process diagram showing how the system works"
          />
        )}

        {/* Steps Grid */}
        <div className="mt-12 grid gap-10 md:grid-cols-2 lg:grid-cols-4">
          {steps.map((step, idx) => (
            <div
              key={idx}
              className={cn(
                'relative rounded-2xl border border-border bg-card/60 p-6 backdrop-blur supports-[backdrop-filter]:bg-background/60',
                'animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none',
                idx === 0 && 'delay-75',
                idx === 1 && 'delay-150',
                idx === 2 && 'delay-200',
                idx === 3 && 'delay-300',
                // Connector arrow (large screens only, tiles 1-3)
                idx < 3 &&
                  "lg:after:absolute lg:after:right-[-26px] lg:after:top-1/2 lg:after:-translate-y-1/2 lg:after:h-3 lg:after:w-3 lg:after:rotate-45 lg:after:rounded-sm lg:after:bg-border lg:after:content-['']",
              )}
            >
              {/* Icon Plate */}
              <div className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 text-foreground">
                {step.icon}
              </div>

              {/* Step Meta */}
              <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-primary">{step.step}</div>

              {/* Title */}
              <h3 className="mb-3 text-lg font-semibold text-foreground">{step.title}</h3>

              {/* Body */}
              <p className="text-sm leading-relaxed text-muted-foreground">{step.body}</p>

              {/* Code Snippet */}
              {step.snippet && (
                <div className="mt-4">
                  <CodeSnippet variant="block" className="tabular-nums text-xs">
                    {step.snippet}
                  </CodeSnippet>
                </div>
              )}

              {/* Checklist */}
              {step.checklist && (
                <ul className="mt-4 space-y-2">
                  {step.checklist.map((item, i) => (
                    <li key={i} className="flex items-center gap-2 text-sm">
                      <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                      <span className="text-muted-foreground">{item}</span>
                    </li>
                  ))}
                </ul>
              )}

              {/* Success Note */}
              {step.successNote && (
                <div className="mt-4 rounded-lg border border-emerald-400/30 bg-emerald-400/10 p-3 animate-in fade-in delay-200 motion-reduce:animate-none">
                  <div className="text-xs font-medium text-emerald-400">{step.successNote}</div>
                </div>
              )}

              {/* Stats Grid */}
              {step.stats && (
                <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                  {step.stats.map((stat, i) => (
                    <div key={i} className="rounded-md border border-border bg-background/60 p-2">
                      <div className="text-muted-foreground">{stat.label}</div>
                      <div className="font-medium text-foreground">{stat.value}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Progress Summary */}
        {avgTime && (
          <div className="mt-10 text-center animate-in fade-in delay-200 motion-reduce:animate-none">
            <div className="inline-flex items-center gap-2 text-lg text-muted-foreground">
              <span>Average setup time:</span>
              {/* Progress bar */}
              <div className="inline-flex h-1.5 w-24 items-center rounded bg-primary/20">
                <div className="h-full w-[70%] rounded bg-primary" />
              </div>
              <span className="font-bold text-primary">{avgTime}</span>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
