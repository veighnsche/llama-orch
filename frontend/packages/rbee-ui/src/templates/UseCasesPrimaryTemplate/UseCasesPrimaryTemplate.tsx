'use client'

import { UseCaseCard } from '@rbee/ui/molecules'
import type { ReactNode } from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export interface UseCasePrimaryItem {
  icon: ReactNode
  color: 'chart-2' | 'primary' | 'chart-3' | 'chart-4'
  title: string
  scenario: string
  solution: string
  outcome: string
  highlights: string[]
  anchor?: string
  badge?: string
}

export interface UseCasePrimaryFilterItem {
  label: string
  anchor: string
}

export interface UseCasesPrimaryTemplateProps {
  /** Eyebrow text above the hero image */
  eyebrow: string
  /** Hero image/icon component */
  heroImage: ReactNode
  /** Aria label for the hero image */
  heroImageAriaLabel: string
  /** Filter buttons for navigation */
  filters: UseCasePrimaryFilterItem[]
  /** Array of use case items */
  useCases: UseCasePrimaryItem[]
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * Primary use cases section with filtering and grid layout
 *
 * @example
 * ```tsx
 * import { UsecasesGridDark } from '@rbee/ui/icons'
 * import { Laptop, Users, Home } from 'lucide-react'
 *
 * <UseCasesPrimaryTemplate
 *   eyebrow="OpenAI-compatible • Your GPUs • Zero API Fees"
 *   heroImage={<UsecasesGridDark size="100%" className="w-full h-auto" />}
 *   heroImageAriaLabel="Dark grid of LLM use cases"
 *   filters={[
 *     { label: 'All', anchor: '#use-cases' },
 *     { label: 'Solo', anchor: '#developers' }
 *   ]}
 *   useCases={[
 *     {
 *       icon: <Laptop className="size-6" />,
 *       color: 'chart-2',
 *       title: 'Solo Developer',
 *       scenario: '...',
 *       solution: '...',
 *       outcome: '...',
 *       highlights: ['$0 inference', 'Private by default']
 *     }
 *   ]}
 * />
 * ```
 */
export function UseCasesPrimaryTemplate({
  eyebrow,
  heroImage,
  heroImageAriaLabel,
  filters,
  useCases,
}: UseCasesPrimaryTemplateProps) {
  const handleFilterClick = (anchor: string) => {
    const element = document.querySelector(anchor)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <>
      {/* Header block with eyebrow */}
      <div className="max-w-6xl mx-auto mb-8 animate-in fade-in duration-500">
        <p className="text-center text-sm text-muted-foreground mb-6">{eyebrow}</p>

        {/* Hero strip image */}
        <div className="relative overflow-hidden rounded border/60 mb-8">{heroImage}</div>

        {/* Filter pills */}
        <nav
          aria-label="Filter use cases"
          className="flex flex-wrap items-center justify-center gap-2 animate-in slide-in-from-top-2 duration-500 delay-100"
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

      {/* Responsive grid: 1 col mobile, 2 cols tablet+ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8 max-w-6xl mx-auto">
        {useCases.map((useCase, index) => (
          <UseCaseCard key={useCase.title} {...useCase} style={{ animationDelay: `${index * 60}ms` }} />
        ))}
      </div>
    </>
  )
}
