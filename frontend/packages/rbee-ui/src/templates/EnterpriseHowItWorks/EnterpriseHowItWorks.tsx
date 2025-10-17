import { StepCard } from '@rbee/ui/molecules'
import { TimelineCard } from '@rbee/ui/organisms'
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

export type EnterpriseHowItWorksProps = {
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

export function EnterpriseHowItWorks({
  id,
  backgroundImage,
  deploymentSteps,
  timeline,
}: EnterpriseHowItWorksProps) {
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
            <TimelineCard
              heading={timeline.heading}
              description={timeline.description}
              progress={25}
              weeks={timeline.weeks}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
