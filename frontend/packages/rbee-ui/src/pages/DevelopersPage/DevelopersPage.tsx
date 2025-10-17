'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  CodeExamplesTemplate,
  CTATemplate,
  DevelopersHeroTemplate,
  EmailCapture,
  FeaturesTabs,
  HowItWorks,
  PricingTemplate,
  ProblemTemplate,
  SolutionTemplate,
  TestimonialsTemplate,
  UseCasesTemplate,
} from '@rbee/ui/templates'
import {
  codeExamplesContainerProps,
  codeExamplesProps,
  coreFeatureTabsContainerProps,
  coreFeatureTabsProps,
  ctaTemplateProps,
  developersEmailCaptureContainerProps,
  developersEmailCaptureProps,
  developersHeroProps,
  developersPricingTemplateContainerProps,
  developersPricingTemplateProps,
  howItWorksContainerProps,
  howItWorksProps,
  problemTemplateContainerProps,
  problemTemplateProps,
  solutionTemplateContainerProps,
  solutionTemplateProps,
  testimonialsTemplateContainerProps,
  testimonialsTemplateProps,
  useCasesTemplateContainerProps,
  useCasesTemplateProps,
} from './DevelopersPageProps'

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
      <TemplateContainer {...developersEmailCaptureContainerProps}>
        <EmailCapture {...developersEmailCaptureProps} />
      </TemplateContainer>

      <TemplateContainer {...problemTemplateContainerProps}>
        <ProblemTemplate {...problemTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...solutionTemplateContainerProps}>
        <SolutionTemplate {...solutionTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...howItWorksContainerProps}>
        <HowItWorks {...howItWorksProps} />
      </TemplateContainer>

      <TemplateContainer {...coreFeatureTabsContainerProps}>
        <FeaturesTabs {...coreFeatureTabsProps} />
      </TemplateContainer>

      <TemplateContainer {...useCasesTemplateContainerProps}>
        <UseCasesTemplate {...useCasesTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...codeExamplesContainerProps}>
        <CodeExamplesTemplate {...codeExamplesProps} />
      </TemplateContainer>

      <TemplateContainer {...developersPricingTemplateContainerProps}>
        <PricingTemplate {...developersPricingTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...testimonialsTemplateContainerProps}>
        <TestimonialsTemplate {...testimonialsTemplateProps} />
      </TemplateContainer>

      <CTATemplate {...ctaTemplateProps} />
    </main>
  )
}
