import { type Sector, TESTIMONIALS } from '@rbee/ui/data/testimonials'
import { StatsGrid, TestimonialCard } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

export type ProvidersTestimonialsStat = {
  icon: React.ReactNode
  value: string
  label: string
}

export type ProvidersTestimonialsTemplateProps = {
  sectorFilter: Sector | Sector[]
  stats: ProvidersTestimonialsStat[]
  limit?: number
  headingId?: string
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
export function ProvidersTestimonialsTemplate({
  sectorFilter,
  stats,
  limit,
  headingId = 'testimonials-h2',
}: ProvidersTestimonialsTemplateProps) {
  // Filter testimonials by sector
  const filteredTestimonials = TESTIMONIALS.filter((t: any) => {
    if (!sectorFilter) return true
    if (Array.isArray(sectorFilter)) {
      return sectorFilter.includes(t.sector)
    }
    return t.sector === sectorFilter
  }).slice(0, limit)

  return (
    <div>
      {/* Testimonials Rail */}
      <section aria-labelledby={headingId} className="mb-12">
        <div
          className={cn(
            'animate-in fade-in-50',
            '-mx-6 flex snap-x snap-mandatory gap-6 overflow-x-auto px-6 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring md:mx-0 md:grid md:grid-cols-3 md:snap-none md:overflow-visible md:px-0',
          )}
          tabIndex={0}
          role="region"
          aria-label="Testimonials carousel (scroll horizontally)"
        >
          <span className="sr-only">Use arrow keys or swipe to navigate testimonials</span>
          {filteredTestimonials.map((testimonial: any) => (
            <div key={testimonial.id} className="min-w-[85%] snap-center md:min-w-0">
              <TestimonialCard
                name={testimonial.name}
                role={testimonial.role}
                quote={testimonial.quote}
                avatar={testimonial.avatar}
                company={testimonial.org ? { name: testimonial.org } : undefined}
                rating={testimonial.rating}
                highlight={testimonial.payout}
                verified={!!testimonial.payout}
              />
            </div>
          ))}
        </div>
      </section>

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
