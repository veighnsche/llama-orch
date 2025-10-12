import { PricingHero } from "@/components/pricing/pricing-hero"
import { PricingTiers } from "@/components/pricing/pricing-tiers"
import { PricingComparison } from "@/components/pricing/pricing-comparison"
import { PricingFAQ } from "@/components/pricing/pricing-faq"
import { EmailCapture } from "@/components/email-capture"

export default function PricingPage() {
  return (
    <div className="pt-16">
      <PricingHero />
      <PricingTiers />
      <PricingComparison />
      <PricingFAQ />
      <EmailCapture />
    </div>
  )
}
