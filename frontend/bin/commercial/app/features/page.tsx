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
} from '@rbee/ui/organisms/Features'
import { EmailCapture } from '@rbee/ui/organisms/EmailCapture'

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
