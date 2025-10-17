"use client";

import { TemplateContainer } from "@rbee/ui/molecules";
import { SecurityCard } from "@rbee/ui/organisms";
import {
  CardGridTemplate,
  ComparisonTemplate,
  EmailCapture,
  EnterpriseCompliance,
  EnterpriseCTA,
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
  enterpriseFeaturesData,
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
        <ComparisonTemplate {...enterpriseComparisonProps} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseFeaturesContainerProps}>
        <CardGridTemplate>
          {enterpriseFeaturesData.map((feature, index) => (
            <SecurityCard key={index} {...feature} />
          ))}
        </CardGridTemplate>
      </TemplateContainer>

      <TemplateContainer {...enterpriseTestimonialsContainerProps}>
        <EnterpriseTestimonials {...enterpriseTestimonialsProps} />
      </TemplateContainer>

      <EnterpriseCTA {...enterpriseCTAProps} />
    </main>
  );
}
