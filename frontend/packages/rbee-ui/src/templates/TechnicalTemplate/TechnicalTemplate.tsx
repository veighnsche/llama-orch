import type { ArchitectureHighlight, TechItem } from '@rbee/ui/molecules'
import { ArchitectureHighlights, CoverageProgressBar, TechnologyStack } from '@rbee/ui/molecules'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type CoverageProgress = {
  label: string
  passing: number
  total: number
}

export type ArchitectureDiagram = {
  /** SVG component to render */
  component: React.ComponentType<{ className?: string; 'aria-label'?: string }>
  /** Accessible label for the diagram */
  ariaLabel: string
}

export type TechnologyStackLinks = {
  githubUrl: string
  license: string
  architectureUrl: string
}

/**
 * TechnicalTemplate displays technical architecture details with highlights and tech stack.
 *
 * @example
 * ```tsx
 * <TechnicalTemplate
 *   architectureHighlights={[
 *     { title: 'BDD-Driven', details: ['42/62 scenarios passing'] },
 *   ]}
 *   coverageProgress={{ label: 'BDD Coverage', passing: 42, total: 62 }}
 *   architectureDiagram={{ component: RbeeArch, ariaLabel: 'Architecture diagram' }}
 *   techStack={[
 *     { name: 'Rust', description: 'Performance + safety', ariaLabel: 'Tech: Rust' },
 *   ]}
 *   stackLinks={{ githubUrl: '...', license: 'MIT', architectureUrl: '/docs' }}
 * />
 * ```
 */
export type TechnicalTemplateProps = {
  /** Architecture highlights to display */
  architectureHighlights: ArchitectureHighlight[]
  /** Coverage progress bar data */
  coverageProgress?: CoverageProgress
  /** Architecture diagram configuration */
  architectureDiagram?: ArchitectureDiagram
  /** Technology stack items */
  techStack: TechItem[]
  /** Links for technology stack section */
  stackLinks: TechnologyStackLinks
  /** Custom class name for the root element */
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function TechnicalTemplate({
  architectureHighlights,
  coverageProgress,
  architectureDiagram,
  techStack,
  stackLinks,
  className,
}: TechnicalTemplateProps) {
  const DiagramComponent = architectureDiagram?.component

  return (
    <div className={className}>
      <div className="grid grid-cols-12 gap-6 lg:gap-10 max-w-6xl mx-auto motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-500">
        {/* Left Column: Architecture Highlights + Diagram */}
        <div className="col-span-12 lg:col-span-6 space-y-6">
          {/* Architecture Highlights */}
          <ArchitectureHighlights highlights={architectureHighlights} />

          {/* BDD Coverage Progress Bar */}
          {coverageProgress && (
            <CoverageProgressBar
              label={coverageProgress.label}
              passing={coverageProgress.passing}
              total={coverageProgress.total}
              className="mt-6"
            />
          )}

          {/* Architecture Diagram (Desktop Only) */}
          {DiagramComponent && (
            <DiagramComponent
              className="hidden md:block rounded ring-1 ring-border/60 shadow-sm"
              aria-label={architectureDiagram.ariaLabel}
            />
          )}
        </div>

        {/* Right Column: Technology Stack (Sticky on Large Screens) */}
        <div className="col-span-12 lg:col-span-6 lg:sticky lg:top-20">
          <TechnologyStack
            technologies={techStack}
            githubUrl={stackLinks.githubUrl}
            license={stackLinks.license}
            architectureUrl={stackLinks.architectureUrl}
          />
        </div>
      </div>
    </div>
  )
}
