'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  CTATemplate,
  EmailCapture,
  EnterpriseCompliance,
  EnterpriseHero,
  EnterpriseHowItWorks,
  EnterpriseSecurity,
  FAQTemplate,
  HowItWorks,
  ProblemTemplate,
  SecurityIsolation,
  SolutionTemplate,
  TechnicalTemplate,
} from '@rbee/ui/templates'
import {
  securityArchitectureContainerProps,
  securityArchitectureProps,
  securityCratesContainerProps,
  securityCratesProps,
  securityCTAContainerProps,
  securityCTAProps,
  securityDefenseLayersContainerProps,
  securityDefenseLayersProps,
  securityEmailCaptureContainerProps,
  securityEmailCaptureProps,
  securityFAQsContainerProps,
  securityFAQsProps,
  securityGuaranteesContainerProps,
  securityGuaranteesProps,
  securityHeroProps,
  securityIsolationContainerProps,
  securityIsolationProps,
  securitySDLCContainerProps,
  securitySDLCProps,
  securityThreatModelContainerProps,
  securityThreatModelProps,
  securityVulnerabilityDisclosureContainerProps,
  securityVulnerabilityDisclosureProps,
} from './SecurityPageProps'

export default function SecurityPage() {
  return (
    <main>
      <EnterpriseHero {...securityHeroProps} />

      <TemplateContainer {...securityEmailCaptureContainerProps}>
        <EmailCapture {...securityEmailCaptureProps} />
      </TemplateContainer>

      <TemplateContainer {...securityThreatModelContainerProps}>
        <ProblemTemplate {...securityThreatModelProps} />
      </TemplateContainer>

      <TemplateContainer {...securityDefenseLayersContainerProps}>
        <SolutionTemplate {...securityDefenseLayersProps} />
      </TemplateContainer>

      <TemplateContainer {...securityCratesContainerProps}>
        <EnterpriseSecurity {...securityCratesProps} />
      </TemplateContainer>

      <TemplateContainer {...securityIsolationContainerProps}>
        <SecurityIsolation {...securityIsolationProps} />
      </TemplateContainer>

      <TemplateContainer {...securityGuaranteesContainerProps}>
        <EnterpriseCompliance {...securityGuaranteesProps} />
      </TemplateContainer>

      <TemplateContainer {...securitySDLCContainerProps}>
        <EnterpriseHowItWorks {...securitySDLCProps} />
      </TemplateContainer>

      <TemplateContainer {...securityVulnerabilityDisclosureContainerProps}>
        <HowItWorks {...securityVulnerabilityDisclosureProps} />
      </TemplateContainer>

      <TemplateContainer {...securityArchitectureContainerProps}>
        <TechnicalTemplate {...securityArchitectureProps} />
      </TemplateContainer>

      <TemplateContainer {...securityFAQsContainerProps}>
        <FAQTemplate {...securityFAQsProps} />
      </TemplateContainer>

      <TemplateContainer {...securityCTAContainerProps}>
        <CTATemplate {...securityCTAProps} />
      </TemplateContainer>
    </main>
  )
}
