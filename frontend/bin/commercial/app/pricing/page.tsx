import {
  PricingHero,
  PricingComparison,
  PricingSection,
  FAQSection,
  pricingFaqItems,
  pricingCategories,
  EmailCapture,
} from '@rbee/ui/organisms'

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
