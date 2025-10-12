import { SectionContainer } from "@/components/primitives"

export function PricingFAQ() {
  return (
    <SectionContainer
      title="Pricing FAQs"
      bgVariant="background"
    >
            <div className="max-w-3xl mx-auto space-y-6">
          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-foreground mb-2">Is the free tier really free forever?</h3>
            <p className="text-muted-foreground leading-relaxed">
              Yes. rbee is GPL open source. The Home/Lab tier is completely free with no time limits, no feature
              restrictions, and no hidden costs. You only pay for electricity to run your hardware.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-foreground mb-2">What's the difference between tiers?</h3>
            <p className="text-muted-foreground leading-relaxed">
              All tiers include the full rbee orchestrator with no feature gates. Paid tiers add convenience features
              like Web UI, team collaboration, priority support, and enterprise services. The core AI orchestration is
              identical across all tiers.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-foreground mb-2">Can I upgrade or downgrade anytime?</h3>
            <p className="text-muted-foreground leading-relaxed">
              Yes. You can upgrade to a paid tier anytime to access additional features. You can also downgrade back
              to the free tier without losing your data or configuration.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-foreground mb-2">Do you offer discounts for non-profits?</h3>
            <p className="text-muted-foreground leading-relaxed">
              Yes. We offer 50% discounts for registered non-profits, educational institutions, and open source
              projects. Contact sales for details.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-foreground mb-2">What payment methods do you accept?</h3>
            <p className="text-muted-foreground leading-relaxed">
              We accept credit cards, bank transfers, and purchase orders for enterprise customers. All payments are
              processed securely through Stripe.
            </p>
          </div>

          <div className="bg-card border border-border rounded-lg p-6">
            <h3 className="text-lg font-bold text-foreground mb-2">Is there a trial period?</h3>
            <p className="text-muted-foreground leading-relaxed">
              The Team tier includes a 30-day free trial with full access to all features. No credit card required to
              start. Enterprise customers can request a custom trial period.
            </p>
          </div>
        </div>
      </SectionContainer>
  )
}
