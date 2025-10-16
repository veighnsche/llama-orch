'use client'

import { TemplateContainer } from "@rbee/ui/molecules";
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
} from "@rbee/ui/templates";
import {
  additionalFeaturesGridProps,
  crossNodeOrchestrationProps,
  crossNodeOrchestrationContainerProps,
  errorHandlingProps,
  errorHandlingContainerProps,
  featuresEmailCaptureProps,
  featuresFeaturesTabsProps,
  intelligentModelManagementProps,
  intelligentModelManagementContainerProps,
  multiBackendGpuProps,
  multiBackendGpuContainerProps,
  realTimeProgressProps,
  realTimeProgressContainerProps,
  securityIsolationProps,
  securityIsolationContainerProps,
  additionalFeaturesGridContainerProps,
} from "./FeaturesPageProps";

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
  );
}
