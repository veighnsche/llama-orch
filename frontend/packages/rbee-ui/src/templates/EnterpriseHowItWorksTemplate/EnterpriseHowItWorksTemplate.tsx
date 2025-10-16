import { StepCard } from '@rbee/ui/molecules'
import Image from 'next/image'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type DeploymentStep = {
  index: number
  icon: React.ReactNode
  title: string
  intro: string
  items: string[]
}

export type TimelineWeek = {
  week: string
  phase: string
}

export type EnterpriseHowItWorksTemplateProps = {
  id?: string
  backgroundImage: {
    src: string
    alt: string
  }
  deploymentSteps: DeploymentStep[]
  timeline: {
    heading: string
    description: string
    weeks: TimelineWeek[]
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseHowItWorksTemplate({
  id,
  backgroundImage,
  deploymentSteps,
  timeline,
}: EnterpriseHowItWorksTemplateProps) {
  return (
    <div id={id} className="relative">
      {/* Decorative background illustration */}
      <Image
        src={backgroundImage.src}
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[48rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt={backgroundImage.alt}
        aria-hidden="true"
      />

      <div className="relative z-10">
        {/* Grid: Steps + Timeline */}
        <div className="grid gap-10 lg:grid-cols-[1fr_360px]">
          {/* Steps Rail */}
          <ol className="animate-in fade-in-50 space-y-8 [animation-delay:calc(var(--i)*80ms)]">
            {deploymentSteps.map((step, idx) => (
              <StepCard
                key={step.index}
                index={step.index}
                icon={step.icon}
                title={step.title}
                intro={step.intro}
                items={step.items}
                isLast={idx === deploymentSteps.length - 1}
              />
            ))}
          </ol>

          {/* Sticky Timeline Panel */}
          <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
            <div className="rounded-2xl border border-primary/20 bg-primary/5 p-6">
              <h3 className="mb-2 text-xl font-semibold text-foreground">{timeline.heading}</h3>
              <p className="mb-6 text-sm text-muted-foreground">{timeline.description}</p>

              {/* Progress bar */}
              <div className="mb-6 h-1 rounded bg-border">
                <div className="h-full w-1/4 rounded bg-primary" aria-hidden="true" />
              </div>

              {/* Week chips */}
              <ol className="space-y-3">
                {timeline.weeks.map((week, idx) => (
                  <li
                    key={idx}
                    className="rounded-xl border bg-background px-3 py-2 transition-colors hover:bg-secondary"
                  >
                    <div className="text-sm font-semibold text-primary">{week.week}</div>
                    <div className="text-xs text-muted-foreground">{week.phase}</div>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
