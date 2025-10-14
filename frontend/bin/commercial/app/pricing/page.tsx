import { PricingHero, PricingComparison } from '@rbee/ui/organisms/Pricing'
import { PricingSection } from '@rbee/ui/organisms/PricingSection'
import { FAQSection } from '@rbee/ui/organisms/FaqSection'
import { pricingFaqItems, pricingCategories } from '@rbee/ui/organisms/FaqSection'
import { EmailCapture } from '@rbee/ui/organisms/EmailCapture'

export default function PricingPage() {
  return (
    <>
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
    </>
  )
}
