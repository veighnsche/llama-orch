import { TestimonialsRail } from '@rbee/ui/organisms'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type EnterpriseTestimonialsTemplateProps = {
  heading: string
  description: string
  sectorFilter: string[]
  layout: 'grid' | 'rail'
  showStats: boolean
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseTestimonialsTemplate({
  heading,
  description,
  sectorFilter,
  layout,
  showStats,
}: EnterpriseTestimonialsTemplateProps) {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-16 text-center animate-in fade-in-50 slide-in-from-bottom-2">
          <h2 id="enterprise-testimonials-h2" className="mb-4 text-4xl font-bold text-foreground">
            {heading}
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            {description}
          </p>
        </div>

        {/* Testimonials Rail */}
        <TestimonialsRail
          sectorFilter={sectorFilter}
          layout={layout}
          showStats={showStats}
          headingId="enterprise-testimonials-h2"
        />
      </div>
    </section>
  )
}
