import { AlertTriangle, DollarSign, Lock } from "lucide-react"
import { SectionContainer, FeatureCard } from '@/components/molecules'

export function ProblemSection() {
  return (
    <SectionContainer title="The Hidden Cost of AI Dependency">

      <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <FeatureCard
          icon={AlertTriangle}
          title="Loss of Control"
          description="Providers change models, deprecate APIs, or shut down entirely. Your infrastructure, workflows, and business continuity depend on decisions you can't control."
          iconColor="destructive"
          size="lg"
          className="border-destructive/30 hover:border-destructive/50"
        />
        <FeatureCard
          icon={DollarSign}
          title="Unpredictable Costs"
          description="Pricing changes without warning. Usage scales, costs spiral. What starts affordable becomes unsustainable as your needs grow."
          iconColor="destructive"
          size="lg"
          className="border-destructive/30 hover:border-destructive/50"
        />
        <FeatureCard
          icon={Lock}
          title="Privacy & Compliance Risks"
          description="Sensitive data leaves your network. Compliance requirements clash with cloud dependencies. Audit trails are incomplete. Regulatory exposure grows."
          iconColor="destructive"
          size="lg"
          className="border-destructive/30 hover:border-destructive/50"
        />
      </div>

      <div className="max-w-3xl mx-auto text-center mt-12">
        <p className="text-xl text-muted-foreground leading-relaxed text-balance">
          Whether you're building with AI, monetizing hardware, or ensuring compliance—dependency on external
          providers creates risk you can't afford.
        </p>
      </div>
    </SectionContainer>
  )
}
