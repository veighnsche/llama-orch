"use client";

import { TemplateContainer } from "@rbee/ui/molecules";
import { CoreFeaturesTabs, PricingSection } from "@rbee/ui/organisms";
import {
  CTATemplate,
  DevelopersCodeExamplesTemplate,
  DevelopersHeroTemplate,
  EmailCapture,
  HowItWorks,
  ProblemTemplate,
  SolutionTemplate,
  TestimonialsTemplate,
  UseCasesTemplate,
} from "@rbee/ui/templates";
import {
  coreFeatureTabsProps,
  ctaTemplateProps,
  developersCodeExamplesProps,
  developersEmailCaptureProps,
  developersHeroProps,
  howItWorksContainerProps,
  howItWorksProps,
  developersPricingSectionProps,
  problemTemplateContainerProps,
  problemTemplateProps,
  solutionTemplateContainerProps,
  solutionTemplateProps,
  testimonialsTemplateContainerProps,
  testimonialsTemplateProps,
  useCasesTemplateContainerProps,
  useCasesTemplateProps,
} from "./DevelopersPageProps";

// ============================================================================
// Props Objects
// ============================================================================
// All props imported from single consolidated file: DevelopersPageProps.tsx
// Includes props for all sections on the Developers page
// ============================================================================

export default function DevelopersPage() {
  return (
    <main>
      <DevelopersHeroTemplate {...developersHeroProps} />
      <EmailCapture {...developersEmailCaptureProps} />

      <TemplateContainer {...problemTemplateContainerProps}>
        <ProblemTemplate {...problemTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...solutionTemplateContainerProps}>
        <SolutionTemplate {...solutionTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...howItWorksContainerProps}>
        <HowItWorks {...howItWorksProps} />
      </TemplateContainer>

      <CoreFeaturesTabs {...coreFeatureTabsProps} />

      <TemplateContainer {...useCasesTemplateContainerProps}>
        <UseCasesTemplate {...useCasesTemplateProps} />
      </TemplateContainer>

      <DevelopersCodeExamplesTemplate {...developersCodeExamplesProps} />
      <PricingSection {...developersPricingSectionProps} />

      <TemplateContainer {...testimonialsTemplateContainerProps}>
        <TestimonialsTemplate {...testimonialsTemplateProps} />
      </TemplateContainer>

      <CTATemplate {...ctaTemplateProps} />
    </main>
  );
}
