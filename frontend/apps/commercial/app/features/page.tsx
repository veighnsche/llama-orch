import {
  FeaturesHero,
  CoreFeaturesTabs,
  CrossNodeOrchestration,
  IntelligentModelManagement,
  MultiBackendGpu,
  ErrorHandling,
  RealTimeProgress,
  SecurityIsolation,
  AdditionalFeaturesGrid,
  EmailCapture,
} from '@rbee/ui/organisms'

export default function FeaturesPage() {
  return (
    <>
      <FeaturesHero />
      <CoreFeaturesTabs />
      <CrossNodeOrchestration />
      <IntelligentModelManagement />
      <MultiBackendGpu />
      <ErrorHandling />
      <RealTimeProgress />
      <SecurityIsolation />
      <AdditionalFeaturesGrid />
      <EmailCapture />
    </>
  )
}
