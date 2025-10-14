import { Users, Cpu, TrendingUp, Star } from 'lucide-react'
import { TestimonialsRail } from '@rbee/ui/organisms/TestimonialsRail'
import { StatsGrid } from '@rbee/ui/molecules/StatsGrid'

export function SocialProofSection() {
  return (
    <section
      id="providers-social-proof"
      aria-labelledby="providers-h2"
      className="relative border-b border-border bg-gradient-to-b from-background to-card px-6 py-24"
    >
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <header className="animate-in fade-in-50 slide-in-from-bottom-2 mb-14 text-center motion-reduce:animate-none">
          <p className="mb-2 text-sm tracking-wide text-primary/70">Provider Stories</p>
          <h2
            id="providers-h2"
            className="mb-3 text-balance text-4xl font-bold text-foreground lg:text-5xl"
          >
            What Real Providers Are Earning
          </h2>
          <p className="mx-auto max-w-2xl text-pretty text-lg text-muted-foreground">
            GPU owners on the rbee marketplace turn idle time into steady payouts—fully self-managed,
            OpenAI-compatible infrastructure.
          </p>
          <p className="mt-3 text-xs text-muted-foreground/70">
            Data from verified providers on rbee; payouts vary by GPU, uptime, and demand.
          </p>
        </header>

        {/* Testimonials Rail */}
        <div className="mb-12">
          <TestimonialsRail
            sectorFilter="provider"
            layout="carousel"
            headingId="providers-h2"
          />
        </div>

        {/* Stats Strip */}
        <StatsGrid
          variant="cards"
          columns={4}
          className="animate-in fade-in-50 slide-in-from-bottom-2 delay-300 motion-reduce:animate-none"
          stats={[
            {
              icon: <Users className="h-5 w-5 text-primary" aria-hidden="true" />,
              value: '500+',
              label: 'Active Providers',
            },
            {
              icon: <Cpu className="h-5 w-5 text-primary" aria-hidden="true" />,
              value: '2,000+',
              label: 'GPUs Earning',
            },
            {
              icon: <TrendingUp className="h-5 w-5 text-primary" aria-hidden="true" />,
              value: '€180K+',
              label: 'Paid to Providers',
            },
            {
              icon: <Star className="h-5 w-5 text-primary" aria-hidden="true" />,
              value: '4.8/5',
              label: 'Average Rating',
            },
          ]}
        />
      </div>
    </section>
  )
}

// Export alias for backward compatibility
export const ProvidersTestimonials = SocialProofSection
