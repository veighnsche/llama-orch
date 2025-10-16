'use client'

import {
  EmailCapture,
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
  ProblemSection,
} from "@rbee/ui/organisms";
import {
  enterpriseEmailCaptureProps,
  enterpriseProblemSectionProps,
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
      <ProblemSection {...enterpriseProblemSectionProps} />
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
