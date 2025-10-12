export function PricingHero() {
  return (
    <section className="py-24 bg-gradient-to-b from-background to-card">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl lg:text-6xl font-bold text-foreground mb-6 text-balance">
            Start Free.
            <br />
            <span className="text-primary">Scale When Ready.</span>
          </h1>
          <p className="text-xl text-muted-foreground leading-relaxed">
            All tiers include the full rbee orchestrator. No feature gates. No artificial limits. Just honest pricing
            for honest infrastructure.
          </p>
        </div>
      </div>
    </section>
  )
}
