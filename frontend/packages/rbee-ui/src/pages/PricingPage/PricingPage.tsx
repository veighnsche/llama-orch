'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
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
  pricingEmailCaptureContainerProps,
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

      <TemplateContainer {...pricingTemplateContainerProps}>
        <PricingTemplate {...pricingTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...pricingComparisonContainerProps}>
        <PricingComparisonTemplate {...pricingComparisonProps} />
      </TemplateContainer>

      <TemplateContainer {...pricingFaqContainerProps}>
        <FAQTemplate {...pricingFaqProps} />
      </TemplateContainer>

      <TemplateContainer {...pricingEmailCaptureContainerProps}>
        <EmailCapture {...pricingEmailCaptureProps} />
      </TemplateContainer>
    </main>
  )
}
