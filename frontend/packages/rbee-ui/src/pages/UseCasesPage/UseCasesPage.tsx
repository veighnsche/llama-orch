'use client'

import { TemplateContainer } from "@rbee/ui/molecules";
import {
  EmailCapture,
  UseCasesHeroTemplate,
  UseCasesIndustryTemplate,
  UseCasesPrimaryTemplate,
} from "@rbee/ui/templates";
import {
  useCasesEmailCaptureProps,
  useCasesHeroProps,
  useCasesIndustryContainerProps,
  useCasesIndustryProps,
  useCasesPrimaryContainerProps,
  useCasesPrimaryProps,
} from "./UseCasesPageProps";

// ============================================================================
// Props Objects
// ============================================================================
// All props imported from single consolidated file: UseCasesPageProps.tsx
// Includes both container props and template props for all sections
// ============================================================================

export default function UseCasesPage() {
  return (
    <main>
      <UseCasesHeroTemplate {...useCasesHeroProps} />
      
      <TemplateContainer {...useCasesPrimaryContainerProps}>
        <UseCasesPrimaryTemplate {...useCasesPrimaryProps} />
      </TemplateContainer>
      
      <TemplateContainer {...useCasesIndustryContainerProps}>
        <UseCasesIndustryTemplate {...useCasesIndustryProps} />
      </TemplateContainer>
      
      <EmailCapture {...useCasesEmailCaptureProps} />
    </main>
  );
}
