import { PricingHero } from '@/components/organisms/Pricing/pricing-hero'
import { PricingSection } from '@/components/organisms/PricingSection/PricingSection'
import { PricingComparison } from '@/components/organisms/Pricing/pricing-comparison'
import { PricingFAQ } from '@/components/organisms/Pricing/pricing-faq'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'

export default function PricingPage() {
  return (
    <div className="pt-16">
      <PricingHero />
      <PricingSection variant="pricing" showKicker={false} showEditorialImage={false} />
      <PricingComparison />
      <PricingFAQ />
      <EmailCapture />
    </div>
  )
}
