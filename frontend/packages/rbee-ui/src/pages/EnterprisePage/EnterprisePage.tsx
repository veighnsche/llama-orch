'use client'

import { TemplateContainer } from "@rbee/ui/molecules";
import {
  EnterpriseComparison,
  EnterpriseCompliance,
  EnterpriseCTA,
  EnterpriseFeatures,
  EnterpriseHero,
  EnterpriseHowItWorks,
  EnterpriseSecurity,
  EnterpriseSolution,
  EnterpriseTestimonials,
  EnterpriseUseCases,
} from "@rbee/ui/organisms";
import {
  EmailCapture,
  ProblemTemplate,
} from "@rbee/ui/templates";
import {
  enterpriseEmailCaptureProps,
  enterpriseProblemTemplateContainerProps,
  enterpriseProblemTemplateProps,
} from "./EnterprisePageProps";

// ============================================================================
// Props Objects
// ============================================================================
// All props imported from single consolidated file: EnterprisePageProps.tsx
// Includes props for all sections on the Enterprise page
// ============================================================================

export default function EnterprisePage() {
  return (
    <main>
      <EnterpriseHero />
      <EmailCapture {...enterpriseEmailCaptureProps} />
      
      <TemplateContainer {...enterpriseProblemTemplateContainerProps}>
        <ProblemTemplate {...enterpriseProblemTemplateProps} />
      </TemplateContainer>
      
      <EnterpriseSolution />
      <EnterpriseCompliance />
      <EnterpriseSecurity />
      <EnterpriseHowItWorks />
      <EnterpriseUseCases />
      <EnterpriseComparison />
      <EnterpriseFeatures />
      <EnterpriseTestimonials />
      <EnterpriseCTA />
    </main>
  );
}
