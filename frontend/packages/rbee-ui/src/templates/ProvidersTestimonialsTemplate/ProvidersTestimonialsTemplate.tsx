import type { Sector } from '@rbee/ui/data/testimonials'
import { StatsGrid } from '@rbee/ui/molecules'
import { TestimonialsRail } from '@rbee/ui/organisms'
import type { LucideIcon } from 'lucide-react'
import * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersTestimonialsStat = {
  icon: LucideIcon
  value: string
  label: string
}

export type ProvidersTestimonialsTemplateProps = {
  sectorFilter: Sector | Sector[]
  stats: ProvidersTestimonialsStat[]
}

// ────────────────────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────────────────────

/**
 * ProvidersTestimonialsTemplate - Social proof section with testimonials and stats
 *
 * @example
 * ```tsx
 * <ProvidersTestimonialsTemplate
 *   sectorFilter="provider"
 *   stats={[...]}
 * />
 * ```
 */
export function ProvidersTestimonialsTemplate({ sectorFilter, stats }: ProvidersTestimonialsTemplateProps) {
  return (
    <div>
      {/* Testimonials Rail */}
      <div className="mb-12">
        <TestimonialsRail sectorFilter={sectorFilter} layout="carousel" />
      </div>

      {/* Stats Strip */}
      <StatsGrid
        variant="cards"
        columns={4}
        className="animate-in fade-in-50 slide-in-from-bottom-2 delay-300 motion-reduce:animate-none"
        stats={stats}
      />
    </div>
  )
}
