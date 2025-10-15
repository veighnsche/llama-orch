import { type Sector, TESTIMONIAL_STATS, TESTIMONIALS } from '@rbee/ui/data/testimonials'
import { StatsGrid, TestimonialCard } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'

export interface TestimonialsRailProps {
  sectorFilter?: Sector | Sector[]
  limit?: number
  layout?: 'grid' | 'carousel'
  showStats?: boolean
  className?: string
  headingId?: string
}

export function TestimonialsRail({
  sectorFilter,
  limit,
  layout = 'grid',
  showStats = false,
  className,
  headingId = 'testimonials-h2',
}: TestimonialsRailProps) {
  // Filter testimonials by sector
  const filteredTestimonials = TESTIMONIALS.filter((t: any) => {
    if (!sectorFilter) return true
    if (Array.isArray(sectorFilter)) {
      return sectorFilter.includes(t.sector)
    }
    return t.sector === sectorFilter
  }).slice(0, limit)

  return (
    <section aria-labelledby={headingId} className={cn('', className)}>
      {/* Testimonials */}
      <div
        className={cn(
          'animate-in fade-in-50',
          layout === 'grid' && 'grid gap-8 md:grid-cols-3',
          layout === 'carousel' &&
            '-mx-6 flex snap-x snap-mandatory gap-6 overflow-x-auto px-6 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring md:mx-0 md:grid md:grid-cols-3 md:snap-none md:overflow-visible md:px-0',
        )}
        tabIndex={layout === 'carousel' ? 0 : undefined}
        role={layout === 'carousel' ? 'region' : undefined}
        aria-label={layout === 'carousel' ? 'Testimonials carousel (scroll horizontally)' : undefined}
      >
        {layout === 'carousel' && <span className="sr-only">Use arrow keys or swipe to navigate testimonials</span>}
        {filteredTestimonials.map((testimonial: any, index: number) => (
          <div key={testimonial.id} className={cn(layout === 'carousel' && 'min-w-[85%] snap-center md:min-w-0')}>
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

      {/* Stats */}
      {showStats && (
        <div className="mt-12 animate-in fade-in-50" style={{ animationDelay: '200ms' }}>
          <StatsGrid
            variant="tiles"
            columns={4}
            stats={TESTIMONIAL_STATS.map((stat: any) => ({
              value: stat.value,
              label: stat.label,
            }))}
          />
        </div>
      )}
    </section>
  )
}
