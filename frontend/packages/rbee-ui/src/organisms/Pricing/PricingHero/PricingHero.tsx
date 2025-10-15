import { Badge, Button } from '@rbee/ui/atoms'
import { PricingScaleVisual } from '@rbee/ui/icons'
import { Sparkles } from 'lucide-react'

export function PricingHero() {
  return (
    <section
      aria-labelledby="pricing-hero-title"
      className="relative overflow-hidden py-24 lg:py-28 bg-gradient-to-b from-background to-card"
    >
      {/* Simple radial glow */}
      <div aria-hidden className="pointer-events-none absolute inset-0 opacity-50">
        <div className="absolute -top-1/3 right-[-20%] h-[60rem] w-[60rem] rounded-full bg-primary/5 blur-3xl" />
      </div>

      <div className="container mx-auto px-4">
        <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
          {/* Left: Text content */}
          <div className="max-w-2xl">
            <Badge variant="secondary" className="mb-4">
              Honest Pricing
            </Badge>

            <h1 id="pricing-hero-title" className="text-5xl lg:text-6xl font-bold text-foreground tracking-tight">
              Start Free.
              <br />
              <span className="text-primary">Scale When Ready.</span>
            </h1>

            <p className="mt-6 text-xl text-muted-foreground leading-relaxed">
              Every tier ships the full rbee orchestrator—no feature gates, no artificial limits. OpenAI-compatible API,
              same power on day one. Pay only when you grow.
            </p>

            {/* Buttons */}
            <div className="mt-8 flex gap-3">
              <Button size="lg">View Plans</Button>
              <Button variant="secondary" size="lg">
                Talk to Sales
              </Button>
            </div>

            {/* Assurance checkmarks */}
            <ul className="mt-6 grid grid-cols-2 gap-3 text-sm text-muted-foreground">
              {[
                'Full orchestrator on every tier',
                'No feature gates or limits',
                'OpenAI-compatible API',
                'Cancel anytime',
              ].map((item) => (
                <li key={item} className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary shrink-0" aria-hidden="true" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Right: Visual illustration */}
          <div className="relative">
            <PricingScaleVisual
              size="100%"
              className="rounded-xl opacity-70"
              aria-label="Illustration showing rbee pricing scales from single-GPU homelab to multi-node server setups with progressive cost tiers"
            />
          </div>
        </div>
      </div>
    </section>
  )
}
