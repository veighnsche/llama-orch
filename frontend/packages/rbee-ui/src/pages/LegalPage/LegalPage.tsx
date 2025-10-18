'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  ComparisonTemplate,
  EmailCapture,
  EnterpriseCTA,
  EnterpriseHero,
  EnterpriseHowItWorks,
  EnterpriseSecurity,
  FAQTemplate,
  ProblemTemplate,
  ProvidersEarnings,
  SolutionTemplate,
  UseCasesTemplate,
} from '@rbee/ui/templates'
import {
  legalComparisonContainerProps,
  legalComparisonProps,
  legalCTAContainerProps,
  legalCTAProps,
  legalEmailCaptureContainerProps,
  legalEmailCaptureProps,
  legalFAQContainerProps,
  legalFAQProps,
  legalHeroProps,
  legalHowItWorksContainerProps,
  legalHowItWorksProps,
  legalProblemTemplateContainerProps,
  legalProblemTemplateProps,
  legalROICalculatorContainerProps,
  legalROICalculatorProps,
  legalSecurityContainerProps,
  legalSecurityProps,
  legalSolutionContainerProps,
  legalSolutionProps,
  legalUseCasesContainerProps,
  legalUseCasesProps,
} from './LegalPageProps'

export default function LegalPage() {
  return (
    <main>
      <EnterpriseHero {...legalHeroProps} />
      <TemplateContainer {...legalEmailCaptureContainerProps}>
        <EmailCapture {...legalEmailCaptureProps} />
      </TemplateContainer>

      <TemplateContainer {...legalProblemTemplateContainerProps}>
        <ProblemTemplate {...legalProblemTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...legalSolutionContainerProps}>
        <SolutionTemplate {...legalSolutionProps} />
      </TemplateContainer>

      <TemplateContainer {...legalUseCasesContainerProps}>
        <UseCasesTemplate {...legalUseCasesProps} />
      </TemplateContainer>

      <TemplateContainer {...legalHowItWorksContainerProps}>
        <EnterpriseHowItWorks {...legalHowItWorksProps} />
      </TemplateContainer>

      <TemplateContainer {...legalSecurityContainerProps}>
        <EnterpriseSecurity {...legalSecurityProps} />
      </TemplateContainer>

      <TemplateContainer {...legalComparisonContainerProps}>
        <ComparisonTemplate {...legalComparisonProps} />
      </TemplateContainer>

      <TemplateContainer {...legalROICalculatorContainerProps}>
        <ProvidersEarnings {...legalROICalculatorProps} />
      </TemplateContainer>

      <TemplateContainer {...legalFAQContainerProps}>
        <FAQTemplate {...legalFAQProps} />
      </TemplateContainer>

      <TemplateContainer {...legalCTAContainerProps}>
        <EnterpriseCTA {...legalCTAProps} />
      </TemplateContainer>
    </main>
  )
}
