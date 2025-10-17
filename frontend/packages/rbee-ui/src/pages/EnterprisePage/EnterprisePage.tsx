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
  EnterpriseUseCases,
  ProblemTemplate,
  SolutionTemplate,
  TestimonialsTemplate,
} from "@rbee/ui/templates";
import {
  enterpriseComparisonContainerProps,
  enterpriseComparisonProps,
  enterpriseComplianceContainerProps,
  enterpriseComplianceProps,
  enterpriseCTAContainerProps,
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
  enterpriseTestimonialsData,
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
        <TestimonialsTemplate {...enterpriseTestimonialsData} />
      </TemplateContainer>

      <TemplateContainer {...enterpriseCTAContainerProps}>
        <EnterpriseCTA {...enterpriseCTAProps} />
      </TemplateContainer>
    </main>
  );
}
