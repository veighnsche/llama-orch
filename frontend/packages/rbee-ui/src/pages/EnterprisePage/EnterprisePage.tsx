"use client";

import { TemplateContainer } from "@rbee/ui/molecules";
import {
  EmailCapture,
  EnterpriseComparison,
  EnterpriseCompliance,
  EnterpriseCTA,
  EnterpriseFeatures,
  EnterpriseHero,
  EnterpriseHowItWorks,
  EnterpriseSecurity,
  EnterpriseTestimonials,
  EnterpriseUseCases,
  ProblemTemplate,
  SolutionTemplate,
} from "@rbee/ui/templates";
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
} from "./EnterprisePageProps";

export default function EnterprisePage() {
  return (
    <main>
      <EnterpriseHero {...enterpriseHeroProps} />
      <EmailCapture {...enterpriseEmailCaptureProps} />

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
        <EnterpriseComparison {...enterpriseComparisonProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseFeaturesContainerProps}>
        <EnterpriseFeatures {...enterpriseFeaturesProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseTestimonialsContainerProps}>
        <EnterpriseTestimonials {...enterpriseTestimonialsProps} />
      </TemplateContainer>

      <EnterpriseCTA {...enterpriseCTAProps} />
    </main>
  );
}
