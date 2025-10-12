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
    <section className="border-b border-slate-800 py-24">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <h2 className="mb-4 text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Trusted by Developers Who Value Independence
          </h2>
        </div>

        <div className="mx-auto mt-16 grid max-w-6xl gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="rounded-lg border border-slate-800 bg-slate-900/50 p-6">
              <div className="mb-4 text-4xl">{testimonial.avatar}</div>
              <p className="mb-4 text-balance leading-relaxed text-slate-300">&quot;{testimonial.quote}&quot;</p>
              <div>
                <div className="font-semibold text-white">{testimonial.author}</div>
                <div className="text-sm text-slate-400">{testimonial.role}</div>
              </div>
            </div>
          ))}
        </div>

        <div className="mx-auto mt-12 grid max-w-4xl gap-8 sm:grid-cols-4">
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-white">1,200+</div>
            <div className="text-sm text-slate-400">GitHub Stars</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-white">500+</div>
            <div className="text-sm text-slate-400">Active Installations</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-white">8,000+</div>
            <div className="text-sm text-slate-400">GPUs Orchestrated</div>
          </div>
          <div className="text-center">
            <div className="mb-2 text-3xl font-bold text-amber-400">€0</div>
            <div className="text-sm text-slate-400">Average Monthly Cost</div>
          </div>
        </div>
      </div>
    </section>
  )
}
