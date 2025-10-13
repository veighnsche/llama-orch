import { Users, Cpu, TrendingUp, Star } from 'lucide-react'
import { TestimonialsRail } from '@/components/organisms/TestimonialsRail/TestimonialsRail'
import { StatCard } from '@/components/molecules/StatCard/StatCard'

const PROVIDER_STATS = [
  { value: '500+', label: 'Active Providers', icon: Users },
  { value: '2,000+', label: 'GPUs Earning', icon: Cpu },
  { value: '€180K+', label: 'Paid to Providers', icon: TrendingUp },
  { value: '4.8/5', label: 'Average Rating', icon: Star },
]

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
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 delay-300 grid gap-6 motion-reduce:animate-none md:grid-cols-4">
          {PROVIDER_STATS.map((stat) => {
            const Icon = stat.icon
            return (
              <div
                key={stat.label}
                className="group rounded-xl border border-border bg-gradient-to-b from-card to-background p-4 text-center transition-all hover:border-primary/30 hover:shadow-sm md:p-5"
              >
                <div className="mb-3 flex justify-center">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                    <Icon className="h-5 w-5 text-primary" aria-hidden="true" />
                  </div>
                </div>
                <StatCard value={stat.value} label={stat.label} size="md" />
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}

// Export alias for backward compatibility
export const ProvidersTestimonials = SocialProofSection
