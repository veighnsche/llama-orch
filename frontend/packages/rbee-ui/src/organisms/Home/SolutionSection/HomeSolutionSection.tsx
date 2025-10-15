import { BeeArchitecture, type BeeTopology, FeatureInfoCard, SectionContainer } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

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
    <SectionContainer
      title={title}
      description={subtitle}
      bgVariant="background"
      paddingY="2xl"
      maxWidth="7xl"
      align="center"
      headingId={id}
      className={cn('border-b border-border', className)}
    >
      {/* Benefits Grid */}
      <div className="mx-auto grid max-w-5xl gap-8 sm:grid-cols-2 lg:grid-cols-4">
        {benefits.map((benefit, idx) => (
          <FeatureInfoCard
            key={idx}
            icon={benefit.icon}
            title={benefit.title}
            body={benefit.body}
            tone="neutral"
            size="sm"
          />
        ))}
      </div>

      {/* Architecture Diagram */}
      <BeeArchitecture topology={topology} />
    </SectionContainer>
  )
}
