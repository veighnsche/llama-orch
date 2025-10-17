'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  ComparisonTemplate,
  EmailCapture,
  EnterpriseCompliance,
  EnterpriseCTA,
  EnterpriseHero,
  EnterpriseHowItWorks,
  EnterpriseSecurity,
  EnterpriseUseCases,
  FAQTemplate,
  ProblemTemplate,
  ProvidersEarnings,
  SolutionTemplate,
} from '@rbee/ui/templates'
import {
  complianceAuditCostCalculatorContainerProps,
  complianceAuditCostCalculatorProps,
  complianceAuditProcessContainerProps,
  complianceAuditProcessProps,
  complianceComparisonContainerProps,
  complianceComparisonProps,
  complianceCTAContainerProps,
  complianceCTAProps,
  complianceEmailCaptureContainerProps,
  complianceEmailCaptureProps,
  complianceFAQContainerProps,
  complianceFAQProps,
  complianceHeroProps,
  complianceProblemTemplateContainerProps,
  complianceProblemTemplateProps,
  complianceSecurityContainerProps,
  complianceSecurityProps,
  complianceSolutionContainerProps,
  complianceSolutionProps,
  complianceStandardsContainerProps,
  complianceStandardsProps,
  complianceUseCasesContainerProps,
  complianceUseCasesProps,
} from './CompliancePageProps'

export default function CompliancePage() {
  return (
    <main>
      <EnterpriseHero {...complianceHeroProps} />
      <TemplateContainer {...complianceEmailCaptureContainerProps}>
        <EmailCapture {...complianceEmailCaptureProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceProblemTemplateContainerProps}>
        <ProblemTemplate {...complianceProblemTemplateProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceSolutionContainerProps}>
        <SolutionTemplate {...complianceSolutionProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceStandardsContainerProps}>
        <EnterpriseCompliance {...complianceStandardsProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceSecurityContainerProps}>
        <EnterpriseSecurity {...complianceSecurityProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceAuditProcessContainerProps}>
        <EnterpriseHowItWorks {...complianceAuditProcessProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceUseCasesContainerProps}>
        <EnterpriseUseCases {...complianceUseCasesProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceComparisonContainerProps}>
        <ComparisonTemplate {...complianceComparisonProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceAuditCostCalculatorContainerProps}>
        <ProvidersEarnings {...complianceAuditCostCalculatorProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceFAQContainerProps}>
        <FAQTemplate {...complianceFAQProps} />
      </TemplateContainer>

      <TemplateContainer {...complianceCTAContainerProps}>
        <EnterpriseCTA {...complianceCTAProps} />
      </TemplateContainer>
    </main>
  )
}
