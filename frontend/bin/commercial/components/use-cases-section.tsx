import { Building, Home, Laptop, Users } from "lucide-react"
import { SectionContainer, FeatureCard } from "@/components/primitives"

export function UseCasesSection() {
  return (
    <SectionContainer title="Built for Those Who Value Independence">

      <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto">
        <FeatureCard
          icon={Laptop}
          title="The Solo Developer"
          description="Scenario: Building a SaaS with AI features. Uses Claude for coding but fears vendor lock-in. Solution: Runs rbee on gaming PC + old workstation. Llama 70B for coding, Stable Diffusion for assets. ✓ $0/month AI costs. Complete control. Never blocked by rate limits."
          iconColor="chart-2"
          size="lg"
          className="bg-secondary"
        />

        <FeatureCard
          icon={Users}
          title="The Small Team"
          description="Scenario: 5-person startup. Spending $500/month on AI APIs. Need to cut costs. Solution: Pools team's hardware. 3 workstations + 2 Macs = 8 GPUs total. Shared rbee cluster. ✓ Saves $6,000/year. Faster inference. GDPR-compliant."
          iconColor="primary"
          size="lg"
          className="bg-secondary"
        />

        <FeatureCard
          icon={Home}
          title="The Homelab Enthusiast"
          description="Scenario: Has 4 GPUs collecting dust. Wants to build AI agents for personal projects. Solution: Runs rbee across homelab. Builds custom AI coder, documentation generator, code reviewer. ✓ Turns idle hardware into productive AI infrastructure."
          iconColor="chart-3"
          size="lg"
          className="bg-secondary"
        />

        <FeatureCard
          icon={Building}
          title="The Enterprise"
          description="Scenario: 50-person dev team. Can't send code to external APIs due to compliance. Solution: Deploys rbee on-premises. 20 GPUs across data center. Custom Rhai routing for compliance. ✓ EU-only routing. Full audit trail. Zero external dependencies."
          iconColor="chart-4"
          size="lg"
          className="bg-secondary"
        />
      </div>
    </SectionContainer>
  )
}
