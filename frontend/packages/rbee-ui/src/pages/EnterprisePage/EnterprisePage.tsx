'use client'

import { SectionContainer } from '@rbee/ui/organisms'
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
  enterpriseProblemSectionContainerProps,
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

      <SectionContainer {...enterpriseProblemSectionContainerProps}>
        <ProblemTemplate {...enterpriseProblemTemplateProps} />
      </SectionContainer>

      <EnterpriseSolutionTemplate {...enterpriseSolutionProps} />

      <SectionContainer {...enterpriseComplianceContainerProps}>
        <EnterpriseComplianceTemplate {...enterpriseComplianceProps} />
      </SectionContainer>

      <SectionContainer {...enterpriseSecurityContainerProps}>
        <EnterpriseSecurityTemplate {...enterpriseSecurityProps} />
      </SectionContainer>

      <SectionContainer {...enterpriseHowItWorksContainerProps}>
        <EnterpriseHowItWorksTemplate {...enterpriseHowItWorksProps} />
      </SectionContainer>

      <SectionContainer {...enterpriseUseCasesContainerProps}>
        <EnterpriseUseCasesTemplate {...enterpriseUseCasesProps} />
      </SectionContainer>

      <SectionContainer {...enterpriseComparisonContainerProps}>
        <EnterpriseComparisonTemplate {...enterpriseComparisonProps} />
      </SectionContainer>

      <SectionContainer {...enterpriseFeaturesContainerProps}>
        <EnterpriseFeaturesTemplate {...enterpriseFeaturesProps} />
      </SectionContainer>

      <SectionContainer {...enterpriseTestimonialsContainerProps}>
        <EnterpriseTestimonialsTemplate {...enterpriseTestimonialsProps} />
      </SectionContainer>

      <EnterpriseCTATemplate {...enterpriseCTAProps} />
    </main>
  )
}
