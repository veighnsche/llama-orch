import { FeaturesHero } from '@/components/organisms/Features/features-hero'
import { CoreFeaturesTabs } from '@/components/organisms/Features/core-features-tabs'
import { CrossNodeOrchestration } from '@/components/organisms/Features/cross-node-orchestration'
import { IntelligentModelManagement } from '@/components/organisms/Features/intelligent-model-management'
import { MultiBackendGpu } from '@/components/organisms/Features/multi-backend-gpu'
import { ErrorHandling } from '@/components/organisms/Features/error-handling'
import { RealTimeProgress } from '@/components/organisms/Features/real-time-progress'
import { SecurityIsolation } from '@/components/organisms/Features/security-isolation'
import { AdditionalFeaturesGrid } from '@/components/organisms/Features/additional-features-grid'
import { EmailCapture } from '@/components/organisms/EmailCapture/EmailCapture'

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
