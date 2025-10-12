import { Button } from "@/components/ui/button"
import { Check } from "lucide-react"

export function PricingSection() {
  return (
    <section className="py-24 bg-background">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-foreground mb-6 text-balance">
            Start Free. Scale When Ready.
          </h2>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          {/* Tier 1: Free */}
          <div className="bg-card border-2 border-border rounded-lg p-8 space-y-6">
            <div>
              <h3 className="text-2xl font-bold text-card-foreground">Home/Lab</h3>
              <div className="mt-4">
                <span className="text-4xl font-bold text-card-foreground">$0</span>
                <span className="text-muted-foreground ml-2">forever</span>
              </div>
            </div>

            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Unlimited GPUs</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">OpenAI-compatible API</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Multi-modal support</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Community support</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Open source</span>
              </li>
            </ul>

            <Button className="w-full bg-transparent" variant="outline">
              Download Now
            </Button>

            <p className="text-sm text-muted-foreground text-center">For solo developers, hobbyists, homelab enthusiasts</p>
          </div>

          {/* Tier 2: Team (Most Popular) */}
          <div className="bg-primary/5 border-2 border-primary rounded-lg p-8 space-y-6 relative">
            <div className="absolute -top-4 left-1/2 -translate-x-1/2">
              <span className="bg-primary text-primary-foreground px-4 py-1 rounded-full text-sm font-medium">Most Popular</span>
            </div>

            <div>
              <h3 className="text-2xl font-bold text-card-foreground">Team</h3>
              <div className="mt-4">
                <span className="text-4xl font-bold text-card-foreground">€99</span>
                <span className="text-muted-foreground ml-2">/month</span>
              </div>
              <p className="text-sm text-muted-foreground mt-1">5-10 developers</p>
            </div>

            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-card-foreground font-medium">Everything in Home/Lab</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Web UI management</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Team collaboration</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Priority support</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Rhai script templates</span>
              </li>
            </ul>

            <Button className="w-full bg-primary hover:bg-primary/90 text-primary-foreground">Start 30-Day Trial</Button>

            <p className="text-sm text-muted-foreground text-center">For small teams, startups</p>
          </div>

          {/* Tier 3: Enterprise */}
          <div className="bg-card border-2 border-border rounded-lg p-8 space-y-6">
            <div>
              <h3 className="text-2xl font-bold text-card-foreground">Enterprise</h3>
              <div className="mt-4">
                <span className="text-4xl font-bold text-card-foreground">Custom</span>
              </div>
              <p className="text-sm text-muted-foreground mt-1">Contact sales</p>
            </div>

            <ul className="space-y-3">
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-card-foreground font-medium">Everything in Team</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Dedicated instances</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Custom SLAs</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">White-label option</span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="h-5 w-5 text-chart-3 flex-shrink-0 mt-0.5" />
                <span className="text-muted-foreground">Enterprise support</span>
              </li>
            </ul>

            <Button className="w-full bg-transparent" variant="outline">
              Contact Sales
            </Button>

            <p className="text-sm text-muted-foreground text-center">For large teams, enterprises</p>
          </div>
        </div>

        <p className="text-center text-muted-foreground mt-12 max-w-2xl mx-auto">
          All tiers include the full rbee orchestrator. No feature gates. No artificial limits.
        </p>
      </div>
    </section>
  )
}
