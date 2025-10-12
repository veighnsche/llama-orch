import { Anchor, DollarSign, Laptop, Shield } from "lucide-react"
import { SectionContainer, FeatureCard, ArchitectureDiagram } from "@/components/primitives"

export function SolutionSection() {
  return (
    <SectionContainer
      title={
        <>
          Your Hardware. Your Models. <span className="text-primary">Your Control.</span>
        </>
      }
      subtitle="rbee orchestrates AI inference across every GPU in your home network—workstations, gaming PCs, Macs—turning idle hardware into a private AI infrastructure."
      bgVariant="secondary"
    >

      {/* Architecture Diagram */}
      <div className="max-w-4xl mx-auto mb-16">
        <ArchitectureDiagram className="shadow-lg" />
      </div>

      {/* Key Benefits */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
        <FeatureCard
          icon={DollarSign}
          title="Zero Ongoing Costs"
          description="Pay only for electricity. No subscriptions. No per-token fees."
          iconColor="chart-3"
        />
        <FeatureCard
          icon={Shield}
          title="Complete Privacy"
          description="Code never leaves your network. GDPR-compliant by default."
          iconColor="chart-2"
        />
        <FeatureCard
          icon={Anchor}
          title="Never Changes"
          description="Models update only when YOU decide. No surprise breakages."
          iconColor="primary"
        />
        <FeatureCard
          icon={Laptop}
          title="Use All Your Hardware"
          description="Orchestrate across CUDA, Metal, CPU. Every GPU contributes."
          iconColor="muted-foreground"
        />
      </div>
    </SectionContainer>
  )
}
