import { TestimonialsRail } from '@rbee/ui/organisms/TestimonialsRail'

export function EnterpriseTestimonials() {
  return (
    <section className="border-b border-border bg-background px-6 py-24">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-16 text-center animate-in fade-in-50 slide-in-from-bottom-2">
          <h2 id="enterprise-testimonials-h2" className="mb-4 text-4xl font-bold text-foreground">
            Trusted by Regulated Industries
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Organizations in highly regulated industries trust rbee for compliance-first AI infrastructure.
          </p>
        </div>

        {/* Testimonials Rail */}
        <TestimonialsRail
          sectorFilter={['finance', 'healthcare', 'legal']}
          layout="grid"
          showStats
          headingId="enterprise-testimonials-h2"
        />
      </div>
    </section>
  )
}
