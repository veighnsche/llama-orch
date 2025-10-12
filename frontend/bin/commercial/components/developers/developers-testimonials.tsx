const testimonials = [
  {
    quote:
      "I was spending $80/month on Claude for coding. Now I run Llama 70B on my gaming PC and old workstation. Same quality, $0 cost. Never going back.",
    author: "Alex K.",
    role: "Solo Developer",
    avatar: "👨‍💻",
  },
  {
    quote:
      "We cut our AI costs from $500/month to zero by pooling our team&apos;s hardware. rbee just works. OpenAI-compatible API means no code changes.",
    author: "Sarah M.",
    role: "CTO at StartupCo",
    avatar: "👩‍💼",
  },
  {
    quote:
      "The cascading shutdown guarantee is a game-changer. No more orphaned processes eating VRAM. Press Ctrl+C and everything shuts down cleanly.",
    author: "Marcus T.",
    role: "DevOps Engineer",
    avatar: "👨‍🔧",
  },
]

export function DevelopersTestimonials() {
  return (
    <section className="border-b border-border py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Trusted by Developers Who Value Independence
          </h2>
        </div>

        <div className="mx-auto mt-16 grid max-w-6xl gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="rounded-lg border border-border bg-card p-6">
              <div className="mb-4 text-4xl">{testimonial.avatar}</div>
              <p className="mb-4 text-balance leading-relaxed text-muted-foreground">&quot;{testimonial.quote}&quot;</p>
              <div>
                <div className="font-semibold text-card-foreground">{testimonial.author}</div>
                <div className="text-sm text-muted-foreground">{testimonial.role}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="mx-auto mt-12 grid max-w-4xl gap-8 sm:grid-cols-4">
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-foreground">1,200+</div>
            <div className="text-sm text-muted-foreground">GitHub Stars</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-foreground">500+</div>
            <div className="text-sm text-muted-foreground">Active Installations</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-foreground">8,000+</div>
            <div className="text-sm text-muted-foreground">GPUs Orchestrated</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-primary">€0</div>
            <div className="text-sm text-muted-foreground">Average Monthly Cost</div>
          </div>
        </div>
      </div>
    </section>
  )
}
