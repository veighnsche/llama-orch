"use client";

import { TemplateContainer } from "@rbee/ui/molecules";
import {
  AdditionalFeaturesGrid,
  CrossNodeOrchestration,
  EmailCapture,
  ErrorHandlingTemplate,
  FeaturesHero,
  FeaturesTabs,
  IntelligentModelManagement,
  MultiBackendGpuTemplate,
  RealTimeProgress,
  SecurityIsolation,
} from "@rbee/ui/templates";
import {
  additionalFeaturesGridContainerProps,
  additionalFeaturesGridProps,
  crossNodeOrchestrationContainerProps,
  crossNodeOrchestrationProps,
  errorHandlingContainerProps,
  errorHandlingProps,
  featuresEmailCaptureContainerProps,
  featuresEmailCaptureProps,
  featuresFeaturesTabsContainerProps,
  featuresFeaturesTabsProps,
  intelligentModelManagementContainerProps,
  intelligentModelManagementProps,
  multiBackendGpuContainerProps,
  multiBackendGpuProps,
  realTimeProgressContainerProps,
  realTimeProgressProps,
  securityIsolationContainerProps,
  securityIsolationProps,
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
      <TemplateContainer {...featuresFeaturesTabsContainerProps}>
        <FeaturesTabs {...featuresFeaturesTabsProps} />
      </TemplateContainer>

      <TemplateContainer {...crossNodeOrchestrationContainerProps}>
        <CrossNodeOrchestration {...crossNodeOrchestrationProps} />
      </TemplateContainer>

      <TemplateContainer {...intelligentModelManagementContainerProps}>
        <IntelligentModelManagement {...intelligentModelManagementProps} />
      </TemplateContainer>

      <TemplateContainer {...multiBackendGpuContainerProps}>
        <MultiBackendGpuTemplate {...multiBackendGpuProps} />
      </TemplateContainer>

      <TemplateContainer {...errorHandlingContainerProps}>
        <ErrorHandlingTemplate {...errorHandlingProps} />
      </TemplateContainer>

      <TemplateContainer {...realTimeProgressContainerProps}>
        <RealTimeProgress {...realTimeProgressProps} />
      </TemplateContainer>

      <TemplateContainer {...securityIsolationContainerProps}>
        <SecurityIsolation {...securityIsolationProps} />
      </TemplateContainer>

      <TemplateContainer {...additionalFeaturesGridContainerProps}>
        <AdditionalFeaturesGrid {...additionalFeaturesGridProps} />
      </TemplateContainer>

      <TemplateContainer {...featuresEmailCaptureContainerProps}>
        <EmailCapture {...featuresEmailCaptureProps} />
      </TemplateContainer>
    </main>
  );
}
