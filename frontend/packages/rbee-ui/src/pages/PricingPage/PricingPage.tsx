'use client'

import { TemplateContainer } from "@rbee/ui/molecules";
import { PricingSection } from "@rbee/ui/organisms";
import {
  EmailCapture,
  FAQTemplate,
  PricingComparisonTemplate,
  PricingHeroTemplate,
} from "@rbee/ui/templates";
import {
  pricingComparisonContainerProps,
  pricingComparisonProps,
  pricingEmailCaptureProps,
  pricingFaqContainerProps,
  pricingFaqProps,
  pricingHeroProps,
  pricingSectionProps,
} from "./PricingPageProps";

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
      <PricingSection {...pricingSectionProps} />
      
      <TemplateContainer {...pricingComparisonContainerProps}>
        <PricingComparisonTemplate {...pricingComparisonProps} />
      </TemplateContainer>
      
      <TemplateContainer {...pricingFaqContainerProps}>
        <FAQTemplate {...pricingFaqProps} />
      </TemplateContainer>
      
      <EmailCapture {...pricingEmailCaptureProps} />
    </main>
  );
}
