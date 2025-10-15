import {
  AdditionalFeaturesGrid,
  CrossNodeOrchestration,
  DefaultCoreFeaturesTabs,
  EmailCapture,
  ErrorHandling,
  FeaturesHero,
  IntelligentModelManagement,
  MultiBackendGpu,
  RealTimeProgress,
  SecurityIsolation,
} from '@rbee/ui/organisms'

export default function FeaturesPage() {
  return (
    <>
      <FeaturesHero />
      <DefaultCoreFeaturesTabs />
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
