import { PricingHero } from '@/components/organisms/Pricing/pricing-hero'
import { PricingSection } from '@/components/organisms/PricingSection/PricingSection'
import { PricingComparison } from '@/components/organisms/Pricing/pricing-comparison'
import { FAQSection } from '@/components/organisms/FaqSection/FaqSection'
import { pricingFaqItems, pricingCategories } from '@/components/organisms/FaqSection/pricing-faqs'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'

export default function PricingPage() {
  return (
    <div className="pt-16">
      <PricingHero />
      <PricingSection variant="pricing" showKicker={false} showEditorialImage={false} />
      <PricingComparison />
      <FAQSection
        title="Pricing FAQs"
        subtitle="Answers on licensing, upgrades, trials, and payments."
        badgeText="Pricing • Plans & Billing"
        categories={pricingCategories}
        faqItems={pricingFaqItems}
        showSupportCard={false}
        jsonLdEnabled={true}
      />
      <EmailCapture />
    </div>
  )
}
