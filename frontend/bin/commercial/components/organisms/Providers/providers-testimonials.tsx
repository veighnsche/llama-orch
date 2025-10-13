import { Users, Cpu, TrendingUp, Star } from 'lucide-react'
import { TestimonialCard } from '@/components/molecules/TestimonialCard/TestimonialCard'
import { StatCard } from '@/components/molecules/StatCard/StatCard'

// Data layer for easy content management
const PROVIDER_TESTIMONIALS = [
  {
    name: 'Marcus T.',
    role: 'Gaming PC Owner',
    payout: '€160/mo',
    rating: 5 as const,
    quote: 'My RTX 4090 used to sit idle. Now it brings in €160/mo—set up in under 10 minutes.',
    avatar: { from: 'primary', to: 'chart-2' },
  },
  {
    name: 'Sarah K.',
    role: 'Homelab Enthusiast',
    payout: '€420/mo',
    rating: 5 as const,
    quote: 'Four homelab GPUs now cover my electricity plus profit—€420/mo combined.',
    avatar: { from: 'chart-1', to: 'chart-3' },
  },
  {
    name: 'David L.',
    role: 'Former Miner',
    payout: '€780/mo',
    rating: 5 as const,
    quote: 'After ETH went PoS, my rig gathered dust. With rbee I average €780/mo—better than mining.',
    avatar: { from: 'chart-2', to: 'chart-4' },
  },
]

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
        <div
          className="-mx-6 mb-12 flex snap-x snap-mandatory gap-6 overflow-x-auto px-6 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring md:mx-0 md:grid md:grid-cols-3 md:snap-none md:overflow-visible md:px-0"
          tabIndex={0}
        >
          {PROVIDER_TESTIMONIALS.map((testimonial, idx) => {
            const delays = ['delay-100', 'delay-200', 'delay-300']
            return (
              <div
                key={testimonial.name}
                className={`animate-in fade-in-50 slide-in-from-bottom-2 min-w-[85%] snap-center motion-reduce:animate-none md:min-w-0 ${delays[idx]}`}
              >
                <TestimonialCard
                  name={testimonial.name}
                  role={`${testimonial.role} • ${testimonial.payout}`}
                  quote={testimonial.quote}
                  avatar={testimonial.avatar}
                  rating={testimonial.rating}
                  verified
                  highlight="Verified payout"
                  className="h-full shadow-sm transition-shadow hover:shadow-md"
                />
              </div>
            )
          })}
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
