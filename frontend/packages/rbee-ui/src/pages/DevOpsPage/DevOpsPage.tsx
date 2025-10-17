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
  ErrorHandlingTemplate,
  FAQTemplate,
  ProblemTemplate,
  RealTimeProgress,
  SolutionTemplate,
} from '@rbee/ui/templates'
import {
  devopsComparisonContainerProps,
  devopsComparisonProps,
  devopsComplianceContainerProps,
  devopsComplianceProps,
  devopsCTAContainerProps,
  devopsCTAProps,
  devopsEmailCaptureContainerProps,
  devopsEmailCaptureProps,
  devopsErrorHandlingContainerProps,
  devopsErrorHandlingProps,
  devopsFAQContainerProps,
  devopsFAQProps,
  devopsHeroProps,
  devopsHowItWorksContainerProps,
  devopsHowItWorksProps,
  devopsProblemContainerProps,
  devopsProblemProps,
  devopsRealTimeProgressContainerProps,
  devopsRealTimeProgressProps,
  devopsSecurityContainerProps,
  devopsSecurityProps,
  devopsSolutionContainerProps,
  devopsSolutionProps,
} from './DevOpsPageProps'

export default function DevOpsPage() {
  return (
    <>
      {/* Hero Section */}
      <EnterpriseHero {...devopsHeroProps} />

      {/* Email Capture */}
      <TemplateContainer {...devopsEmailCaptureContainerProps}>
        <EmailCapture {...devopsEmailCaptureProps} />
      </TemplateContainer>

      {/* Problem Section */}
      <TemplateContainer {...devopsProblemContainerProps}>
        <ProblemTemplate {...devopsProblemProps} />
      </TemplateContainer>

      {/* Solution Section */}
      <TemplateContainer {...devopsSolutionContainerProps}>
        <SolutionTemplate {...devopsSolutionProps} />
      </TemplateContainer>

      {/* How It Works */}
      <TemplateContainer {...devopsHowItWorksContainerProps}>
        <EnterpriseHowItWorks {...devopsHowItWorksProps} />
      </TemplateContainer>

      {/* Operational Features */}
      <TemplateContainer {...devopsSecurityContainerProps}>
        <EnterpriseSecurity {...devopsSecurityProps} />
      </TemplateContainer>

      {/* Error Handling */}
      <TemplateContainer {...devopsErrorHandlingContainerProps}>
        <ErrorHandlingTemplate {...devopsErrorHandlingProps} />
      </TemplateContainer>

      {/* Real-Time Monitoring */}
      <TemplateContainer {...devopsRealTimeProgressContainerProps}>
        <RealTimeProgress {...devopsRealTimeProgressProps} />
      </TemplateContainer>

      {/* Deployment Options Comparison */}
      <TemplateContainer {...devopsComparisonContainerProps}>
        <ComparisonTemplate {...devopsComparisonProps} />
      </TemplateContainer>

      {/* SLAs & Guarantees */}
      <TemplateContainer {...devopsComplianceContainerProps}>
        <EnterpriseCompliance {...devopsComplianceProps} />
      </TemplateContainer>

      {/* FAQ */}
      <TemplateContainer {...devopsFAQContainerProps}>
        <FAQTemplate {...devopsFAQProps} />
      </TemplateContainer>

      {/* CTA */}
      <TemplateContainer {...devopsCTAContainerProps}>
        <EnterpriseCTA {...devopsCTAProps} />
      </TemplateContainer>
    </>
  )
}
