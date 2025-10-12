import { PricingHero } from '@/components/organisms/Pricing/pricing-hero'
import { PricingTiers } from '@/components/organisms/Pricing/pricing-tiers'
import { PricingComparison } from '@/components/organisms/Pricing/pricing-comparison'
import { PricingFAQ } from '@/components/organisms/Pricing/pricing-faq'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'

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
