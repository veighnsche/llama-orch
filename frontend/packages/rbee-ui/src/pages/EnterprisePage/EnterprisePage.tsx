'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  EmailCapture,
  EnterpriseComparisonTemplate,
  EnterpriseComplianceTemplate,
  EnterpriseCTATemplate,
  EnterpriseFeaturesTemplate,
  EnterpriseHeroTemplate,
  EnterpriseHowItWorksTemplate,
  EnterpriseSecurityTemplate,
  EnterpriseSolutionTemplate,
  EnterpriseTestimonialsTemplate,
  EnterpriseUseCasesTemplate,
  ProblemTemplate,
} from '@rbee/ui/templates'
import {
  enterpriseComparisonContainerProps,
  enterpriseComparisonProps,
  enterpriseComplianceContainerProps,
  enterpriseComplianceProps,
  enterpriseCTAProps,
  enterpriseEmailCaptureProps,
  enterpriseFeaturesContainerProps,
  enterpriseFeaturesProps,
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
  enterpriseTestimonialsProps,
  enterpriseUseCasesContainerProps,
  enterpriseUseCasesProps,
} from './EnterprisePageProps'

export default function EnterprisePage() {
  return (
    <main>
      <EnterpriseHeroTemplate {...enterpriseHeroProps} />
      <EmailCapture {...enterpriseEmailCaptureProps} />

      <TemplateContainer {...enterpriseProblemTemplateContainerProps}>
        <ProblemTemplate {...enterpriseProblemTemplateProps} />
      </TemplateContainer>

      <EnterpriseSolutionTemplate {...enterpriseSolutionProps} />

      <TemplateContainer {...enterpriseComplianceContainerProps}>
        <EnterpriseComplianceTemplate {...enterpriseComplianceProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseSecurityContainerProps}>
        <EnterpriseSecurityTemplate {...enterpriseSecurityProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseHowItWorksContainerProps}>
        <EnterpriseHowItWorksTemplate {...enterpriseHowItWorksProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseUseCasesContainerProps}>
        <EnterpriseUseCasesTemplate {...enterpriseUseCasesProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseComparisonContainerProps}>
        <EnterpriseComparisonTemplate {...enterpriseComparisonProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseFeaturesContainerProps}>
        <EnterpriseFeaturesTemplate {...enterpriseFeaturesProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseTestimonialsContainerProps}>
        <EnterpriseTestimonialsTemplate {...enterpriseTestimonialsProps} />
      </TemplateContainer>

      <EnterpriseCTATemplate {...enterpriseCTAProps} />
    </main>
  )
}
