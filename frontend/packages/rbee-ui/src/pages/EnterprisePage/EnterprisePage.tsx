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
  enterpriseEmailCaptureProps,
  enterpriseHeroProps,
  enterpriseProblemTemplateContainerProps,
  enterpriseProblemTemplateProps,
} from './EnterprisePageProps'
import {
  enterpriseComplianceContainerProps,
  enterpriseComplianceProps,
  enterpriseSecurityContainerProps,
  enterpriseSecurityProps,
  enterpriseSolutionContainerProps,
  enterpriseSolutionProps,
} from './EnterprisePagePropsExtended'
import {
  enterpriseComparisonContainerProps,
  enterpriseComparisonProps,
  enterpriseCTAProps,
  enterpriseFeaturesContainerProps,
  enterpriseFeaturesProps,
  enterpriseHowItWorksContainerProps,
  enterpriseHowItWorksProps,
  enterpriseTestimonialsContainerProps,
  enterpriseTestimonialsProps,
  enterpriseUseCasesContainerProps,
  enterpriseUseCasesProps,
} from './EnterprisePagePropsExtended2'

// ============================================================================
// Props Objects
// ============================================================================
// All props imported from three files:
// - EnterprisePageProps.tsx: Hero, Email Capture, Problem Template
// - EnterprisePagePropsExtended.tsx: Solution, Compliance, Security
// - EnterprisePagePropsExtended2.tsx: How It Works, Use Cases, Comparison, Features, Testimonials, CTA
// ============================================================================

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
