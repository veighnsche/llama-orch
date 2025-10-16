'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  AudienceSelector,
  ComparisonTemplate,
  CTATemplate,
  EmailCapture,
  FAQTemplate,
  FeaturesTabs,
  HomeHero,
  HowItWorks,
  PricingTemplate,
  ProblemTemplate,
  SolutionTemplate,
  TechnicalTemplate,
  TestimonialsTemplate,
  UseCasesTemplate,
  WhatIsRbee,
} from '@rbee/ui/templates'
import {
  audienceSelectorContainerProps,
  audienceSelectorProps,
  comparisonTemplateContainerProps,
  comparisonTemplateProps,
  ctaTemplateProps,
  emailCaptureProps,
  faqTemplateContainerProps,
  faqTemplateProps,
  featuresTabsProps,
  homeHeroProps,
  howItWorksContainerProps,
  howItWorksProps,
  pricingTemplateContainerProps,
  pricingTemplateProps,
  problemTemplateContainerProps,
  problemTemplateProps,
  solutionTemplateContainerProps,
  solutionTemplateProps,
  technicalTemplateContainerProps,
  technicalTemplateProps,
  testimonialsTemplateContainerProps,
  testimonialsTemplateProps,
  useCasesTemplateContainerProps,
  useCasesTemplateProps,
  whatIsRbeeContainerProps,
  whatIsRbeeProps,
} from './HomePageProps'
export default function HomePage() {
  return (
    <main>
      <HomeHero {...homeHeroProps} />
      <TemplateContainer {...whatIsRbeeContainerProps}>
        <WhatIsRbee {...whatIsRbeeProps} />
      </TemplateContainer>
      <TemplateContainer {...audienceSelectorContainerProps}>
        <AudienceSelector {...audienceSelectorProps} />
      </TemplateContainer>
      <EmailCapture {...emailCaptureProps} />
      <TemplateContainer {...problemTemplateContainerProps}>
        <ProblemTemplate {...problemTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...solutionTemplateContainerProps}>
        <SolutionTemplate {...solutionTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...howItWorksContainerProps}>
        <HowItWorks {...howItWorksProps} />
      </TemplateContainer>
      <FeaturesTabs {...featuresTabsProps} />
      <TemplateContainer {...useCasesTemplateContainerProps}>
        <UseCasesTemplate {...useCasesTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...comparisonTemplateContainerProps}>
        <ComparisonTemplate {...comparisonTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...pricingTemplateContainerProps}>
        <PricingTemplate {...pricingTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...testimonialsTemplateContainerProps}>
        <TestimonialsTemplate {...testimonialsTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...technicalTemplateContainerProps}>
        <TechnicalTemplate {...technicalTemplateProps} />
      </TemplateContainer>
      <TemplateContainer {...faqTemplateContainerProps}>
        <FAQTemplate {...faqTemplateProps} />
      </TemplateContainer>
      <CTATemplate {...ctaTemplateProps} />
    </main>
  )
}
