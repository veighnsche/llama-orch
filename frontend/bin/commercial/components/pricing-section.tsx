import { SectionContainer, PricingTier } from "@/components/primitives"

export function PricingSection() {
  return (
    <SectionContainer title="Start Free. Scale When Ready.">

      <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <PricingTier
          title="Home/Lab"
          price="$0"
          period="forever"
          features={[
            "Unlimited GPUs",
            "OpenAI-compatible API",
            "Multi-modal support",
            "Community support",
            "Open source"
          ]}
          ctaText="Download Now"
          ctaVariant="outline"
        />

        <PricingTier
          title="Team"
          price="€99"
          period="/month"
          features={[
            "Everything in Home/Lab",
            "Web UI management",
            "Team collaboration",
            "Priority support",
            "Rhai script templates"
          ]}
          ctaText="Start 30-Day Trial"
          highlighted
          badge="Most Popular"
        />

        <PricingTier
          title="Enterprise"
          price="Custom"
          features={[
            "Everything in Team",
            "Dedicated instances",
            "Custom SLAs",
            "White-label option",
            "Enterprise support"
          ]}
          ctaText="Contact Sales"
          ctaVariant="outline"
        />
      </div>

      <p className="text-center text-muted-foreground mt-12 max-w-2xl mx-auto">
        All tiers include the full rbee orchestrator. No feature gates. No artificial limits.
      </p>
    </SectionContainer>
  )
}
