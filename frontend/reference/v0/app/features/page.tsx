import { FeaturesHero } from "@/components/features/features-hero"
import { CoreFeaturesTabs } from "@/components/features/core-features-tabs"
import { CrossNodeOrchestration } from "@/components/features/cross-node-orchestration"
import { IntelligentModelManagement } from "@/components/features/intelligent-model-management"
import { MultiBackendGpu } from "@/components/features/multi-backend-gpu"
import { ErrorHandling } from "@/components/features/error-handling"
import { RealTimeProgress } from "@/components/features/real-time-progress"
import { SecurityIsolation } from "@/components/features/security-isolation"
import { AdditionalFeaturesGrid } from "@/components/features/additional-features-grid"
import { EmailCapture } from "@/components/email-capture"

export default function FeaturesPage() {
  return (
    <div className="pt-16">
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
    </div>
  )
}
