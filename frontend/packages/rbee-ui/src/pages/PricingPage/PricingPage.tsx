'use client'

import { SectionContainer } from '@rbee/ui/organisms'
import {
  EmailCapture,
  FAQTemplate,
  PricingComparisonTemplate,
  PricingHeroTemplate,
  PricingTemplate,
} from '@rbee/ui/templates'
import {
  pricingComparisonContainerProps,
  pricingComparisonProps,
  pricingEmailCaptureProps,
  pricingFaqContainerProps,
  pricingFaqProps,
  pricingHeroProps,
  pricingTemplateContainerProps,
  pricingTemplateProps,
} from './PricingPageProps'

// ============================================================================
// Props Objects
// ============================================================================
// All props imported from single consolidated file: PricingPageProps.tsx
// Includes both container props and template props for all sections
// ============================================================================

export default function PricingPage() {
  return (
    <main>
      <PricingHeroTemplate {...pricingHeroProps} />

      <SectionContainer {...pricingTemplateContainerProps}>
        <PricingTemplate {...pricingTemplateProps} />
      </SectionContainer>

      <SectionContainer {...pricingComparisonContainerProps}>
        <PricingComparisonTemplate {...pricingComparisonProps} />
      </SectionContainer>

      <SectionContainer {...pricingFaqContainerProps}>
        <FAQTemplate {...pricingFaqProps} />
      </SectionContainer>

      <EmailCapture {...pricingEmailCaptureProps} />
    </main>
  )
}
