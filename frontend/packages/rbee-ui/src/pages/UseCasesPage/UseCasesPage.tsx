'use client'

import { SectionContainer } from '@rbee/ui/organisms'
import {
  EmailCapture,
  UseCasesHeroTemplate,
  UseCasesIndustryTemplate,
  UseCasesPrimaryTemplate,
} from '@rbee/ui/templates'
import {
  useCasesEmailCaptureProps,
  useCasesHeroProps,
  useCasesIndustryContainerProps,
  useCasesIndustryProps,
  useCasesPrimaryContainerProps,
  useCasesPrimaryProps,
} from './UseCasesPageProps'

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

      <SectionContainer {...useCasesPrimaryContainerProps}>
        <UseCasesPrimaryTemplate {...useCasesPrimaryProps} />
      </SectionContainer>

      <SectionContainer {...useCasesIndustryContainerProps}>
        <UseCasesIndustryTemplate {...useCasesIndustryProps} />
      </SectionContainer>

      <EmailCapture {...useCasesEmailCaptureProps} />
    </main>
  )
}
