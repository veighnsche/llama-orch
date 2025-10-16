'use client'

import { TemplateContainer } from '@rbee/ui/molecules'
import {
  AdditionalFeaturesGridTemplate,
  CrossNodeOrchestrationTemplate,
  EmailCapture,
  ErrorHandlingTemplate,
  FeaturesHero,
  FeaturesTabs,
  IntelligentModelManagementTemplate,
  MultiBackendGpuTemplate,
  RealTimeProgressTemplate,
  SecurityIsolationTemplate,
} from '@rbee/ui/templates'
import {
  additionalFeaturesGridContainerProps,
  additionalFeaturesGridProps,
  crossNodeOrchestrationContainerProps,
  crossNodeOrchestrationProps,
  errorHandlingContainerProps,
  errorHandlingProps,
  featuresEmailCaptureProps,
  featuresFeaturesTabsProps,
  intelligentModelManagementContainerProps,
  intelligentModelManagementProps,
  multiBackendGpuContainerProps,
  multiBackendGpuProps,
  realTimeProgressContainerProps,
  realTimeProgressProps,
  securityIsolationContainerProps,
  securityIsolationProps,
} from './FeaturesPageProps'

// ============================================================================
// Props Objects
// ============================================================================
// All props imported from single consolidated file: FeaturesPageProps.tsx
// Includes both container props and template props for all sections
// ============================================================================

export default function FeaturesPage() {
  return (
    <main>
      <FeaturesHero />
      <FeaturesTabs {...featuresFeaturesTabsProps} />

      <TemplateContainer {...crossNodeOrchestrationContainerProps}>
        <CrossNodeOrchestrationTemplate {...crossNodeOrchestrationProps} />
      </TemplateContainer>

      <TemplateContainer {...intelligentModelManagementContainerProps}>
        <IntelligentModelManagementTemplate {...intelligentModelManagementProps} />
      </TemplateContainer>

      <TemplateContainer {...multiBackendGpuContainerProps}>
        <MultiBackendGpuTemplate {...multiBackendGpuProps} />
      </TemplateContainer>

      <TemplateContainer {...errorHandlingContainerProps}>
        <ErrorHandlingTemplate {...errorHandlingProps} />
      </TemplateContainer>

      <TemplateContainer {...realTimeProgressContainerProps}>
        <RealTimeProgressTemplate {...realTimeProgressProps} />
      </TemplateContainer>

      <TemplateContainer {...securityIsolationContainerProps}>
        <SecurityIsolationTemplate {...securityIsolationProps} />
      </TemplateContainer>

      <TemplateContainer {...additionalFeaturesGridContainerProps}>
        <AdditionalFeaturesGridTemplate {...additionalFeaturesGridProps} />
      </TemplateContainer>

      <EmailCapture {...featuresEmailCaptureProps} />
    </main>
  )
}
