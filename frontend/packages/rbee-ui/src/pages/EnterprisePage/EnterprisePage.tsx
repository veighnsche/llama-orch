'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  CardGridTemplate,
  ComparisonTemplate,
  EmailCapture,
  EnterpriseCompliance,
  EnterpriseCTA,
  EnterpriseHero,
  EnterpriseHowItWorks,
  EnterpriseSecurity,
  EnterpriseUseCases,
  ProblemTemplate,
  SolutionTemplate,
  TestimonialsTemplate,
} from '@rbee/ui/templates'
import {
  enterpriseComparisonContainerProps,
  enterpriseComparisonProps,
  enterpriseComplianceContainerProps,
  enterpriseComplianceProps,
  enterpriseCTAContainerProps,
  enterpriseCTAProps,
  enterpriseEmailCaptureContainerProps,
  enterpriseEmailCaptureProps,
  enterpriseFeaturesContainerProps,
  enterpriseFeaturesGridProps,
  enterpriseHeroProps,
  enterpriseHowItWorksContainerProps,
  enterpriseHowItWorksProps,
  enterpriseProblemTemplateContainerProps,
  enterpriseProblemTemplateProps,
  enterpriseSecurityContainerProps,
  enterpriseSecurityProps,
  enterpriseSolutionContainerProps,
  enterpriseSolutionProps,
  enterpriseTestimonialsContainerProps,
  enterpriseTestimonialsData,
  enterpriseUseCasesContainerProps,
  enterpriseUseCasesProps,
} from './EnterprisePageProps'

export default function EnterprisePage() {
  return (
    <main>
      <EnterpriseHero {...enterpriseHeroProps} />
      <TemplateContainer {...enterpriseEmailCaptureContainerProps}>
        <EmailCapture {...enterpriseEmailCaptureProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseProblemTemplateContainerProps}>
        <ProblemTemplate {...enterpriseProblemTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseSolutionContainerProps}>
        <SolutionTemplate {...enterpriseSolutionProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseComplianceContainerProps}>
        <EnterpriseCompliance {...enterpriseComplianceProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseSecurityContainerProps}>
        <EnterpriseSecurity {...enterpriseSecurityProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseHowItWorksContainerProps}>
        <EnterpriseHowItWorks {...enterpriseHowItWorksProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseUseCasesContainerProps}>
        <EnterpriseUseCases {...enterpriseUseCasesProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseComparisonContainerProps}>
        <ComparisonTemplate {...enterpriseComparisonProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseFeaturesContainerProps}>
        <CardGridTemplate {...enterpriseFeaturesGridProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseTestimonialsContainerProps}>
        <TestimonialsTemplate {...enterpriseTestimonialsData} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseCTAContainerProps}>
        <EnterpriseCTA {...enterpriseCTAProps} />
      </TemplateContainer>
    </main>
  )
}
