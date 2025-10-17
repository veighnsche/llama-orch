'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  AdditionalFeaturesGrid,
  CTATemplate,
  EmailCapture,
  FAQTemplate,
  FeaturesTabs,
  HeroTemplate,
  HowItWorks,
  ProblemTemplate,
  SolutionTemplate,
  TechnicalTemplate,
  UseCasesTemplate,
} from '@rbee/ui/templates'
import {
  ctaProps,
  determinismContainerProps,
  determinismProps,
  emailCaptureContainerProps,
  emailCaptureProps,
  faqContainerProps,
  faqProps,
  heroContainerProps,
  heroProps,
  multiModalContainerProps,
  multiModalProps,
  problemContainerProps,
  problemProps,
  solutionContainerProps,
  solutionProps,
  technicalContainerProps,
  technicalProps,
  useCasesContainerProps,
  useCasesProps,
  workflowContainerProps,
  workflowProps,
} from './ResearchPageProps'

export default function ResearchPage() {
  return (
    <>
      {/* Hero */}
      <TemplateContainer {...heroContainerProps}>
        <HeroTemplate {...heroProps} />
      </TemplateContainer>

      {/* Email Capture */}
      <TemplateContainer {...emailCaptureContainerProps}>
        <EmailCapture {...emailCaptureProps} />
      </TemplateContainer>

      {/* Problem: Research Challenges */}
      <TemplateContainer {...problemContainerProps}>
        <ProblemTemplate {...problemProps} />
      </TemplateContainer>

      {/* Solution: Reproducibility Features */}
      <TemplateContainer {...solutionContainerProps}>
        <SolutionTemplate {...solutionProps} />
      </TemplateContainer>

      {/* Multi-Modal Support */}
      <TemplateContainer {...multiModalContainerProps}>
        <FeaturesTabs {...multiModalProps} />
      </TemplateContainer>

      {/* Research Workflow */}
      <TemplateContainer {...workflowContainerProps}>
        <HowItWorks {...workflowProps} />
      </TemplateContainer>

      {/* Determinism Suite */}
      <TemplateContainer {...determinismContainerProps}>
        <AdditionalFeaturesGrid {...determinismProps} />
      </TemplateContainer>

      {/* Academic Use Cases */}
      <TemplateContainer {...useCasesContainerProps}>
        <UseCasesTemplate {...useCasesProps} />
      </TemplateContainer>

      {/* Technical Deep-Dive */}
      <TemplateContainer {...technicalContainerProps}>
        <TechnicalTemplate {...technicalProps} />
      </TemplateContainer>

      {/* FAQ */}
      <TemplateContainer {...faqContainerProps}>
        <FAQTemplate {...faqProps} />
      </TemplateContainer>

      {/* Final CTA */}
      <CTATemplate {...ctaProps} />
    </>
  )
}
