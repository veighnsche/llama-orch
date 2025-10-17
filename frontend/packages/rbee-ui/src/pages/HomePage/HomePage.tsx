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
  emailCaptureContainerProps,
  emailCaptureProps,
  faqTemplateContainerProps,
  faqTemplateProps,
  featuresTabsContainerProps,
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
      <TemplateContainer {...emailCaptureContainerProps}>
        <EmailCapture {...emailCaptureProps} />
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
      <TemplateContainer {...featuresTabsContainerProps}>
        <FeaturesTabs {...featuresTabsProps} />
      </TemplateContainer>
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
