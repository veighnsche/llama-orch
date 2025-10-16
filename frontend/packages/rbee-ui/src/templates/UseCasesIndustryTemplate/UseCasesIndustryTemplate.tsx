'use client'

import { IndustryCard, type IndustryCardProps } from '@rbee/ui/molecules'
import type { ReactNode } from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface UseCaseIndustryFilterItem {
  label: string
  anchor: string
}

export interface UseCasesIndustryTemplateProps {
  /** Eyebrow text above the hero image */
  eyebrow: string
  /** Hero image/icon component */
  heroImage: ReactNode
  /** Aria label for the hero image */
  heroImageAriaLabel: string
  /** Filter buttons for navigation */
  filters: UseCaseIndustryFilterItem[]
  /** Array of industry items */
  industries: IndustryCardProps[]
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * Industry-specific use cases section with filtering and grid layout
 *
 * @example
 * ```tsx
 * import { IndustriesHero } from '@rbee/ui/icons'
 * import { Banknote, Heart } from 'lucide-react'
 *
 * <UseCasesIndustryTemplate
 *   eyebrow="Regulated sectors · Private-by-design"
 *   heroImage={<IndustriesHero size="100%" className="w-full h-auto" />}
 *   heroImageAriaLabel="Visual representation of various industry sectors"
 *   filters={[
 *     { label: 'All', anchor: '#architecture' },
 *     { label: 'Finance', anchor: '#finance' }
 *   ]}
 *   industries={[
 *     {
 *       title: 'Financial Services',
 *       icon: <Banknote className="size-6" />,
 *       color: 'primary',
 *       badge: 'GDPR',
 *       copy: 'GDPR-ready with audit trails...',
 *       anchor: 'finance'
 *     }
 *   ]}
 * />
 * ```
 */
export function UseCasesIndustryTemplate({
  eyebrow,
  heroImage,
  heroImageAriaLabel,
  filters,
  industries,
}: UseCasesIndustryTemplateProps) {
  const handleFilterClick = (anchor: string) => {
    const element = document.querySelector(anchor)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <>
      {/* Header block */}
      <div className="max-w-6xl mx-auto mb-8 animate-in fade-in duration-400">
        <p className="text-center text-sm text-muted-foreground mb-6">{eyebrow}</p>

        {/* Hero banner */}
        <div className="overflow-hidden rounded-lg border/60 mb-8">{heroImage}</div>

        {/* Filter pills */}
        <nav
          aria-label="Filter industries"
          className="flex flex-wrap items-center justify-center gap-2 animate-in slide-in-from-top-2 duration-400 delay-75"
        >
          {filters.map((filter) => (
            <button
              key={filter.label}
              onClick={() => handleFilterClick(filter.anchor)}
              className="inline-flex items-center rounded-full border/60 bg-card px-4 py-2 text-sm font-medium text-foreground hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
            >
              {filter.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Responsive grid: 1 col mobile, 2 cols tablet, 3 cols desktop */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 lg:gap-8 max-w-6xl mx-auto">
        {industries.map((industry, index) => (
          <IndustryCard key={industry.title} {...industry} style={{ animationDelay: `${index * 60}ms` }} />
        ))}
      </div>
    </>
  )
}
