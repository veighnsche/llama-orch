import { SectionContainer, StatCard, TestimonialCard } from '@/components/molecules'

export function SocialProofSection() {
  return (
    <SectionContainer 
      title="Trusted by Developers Who Value Independence"
      bgVariant="secondary"
    >

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto mb-16">
        <StatCard value="1,200+" label="GitHub Stars" />
        <StatCard value="500+" label="Active Installations" />
        <StatCard value="8,000+" label="GPUs Orchestrated" />
        <StatCard value="€0" label="Avg Monthly Cost" variant="success" />
      </div>

      {/* Testimonials */}
      <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
        <TestimonialCard
          name="Alex K."
          role="Solo Developer"
          quote="I was spending $80/month on Claude for coding. Now I run Llama 70B on my gaming PC and old workstation. Same quality, $0 cost. Never going back."
          avatar={{ from: "blue-400", to: "blue-600" }}
        />
        <TestimonialCard
          name="Sarah M."
          role="CTO at StartupCo"
          quote="We cut our AI costs from $500/month to zero by pooling our team's hardware. rbee just works. OpenAI-compatible API means no code changes."
          avatar={{ from: "amber-400", to: "amber-600" }}
        />
        <TestimonialCard
          name="Dr. Thomas R."
          role="Research Lab Director"
          quote="GDPR compliance was killing us. rbee let us build AI infrastructure on-premises. EU-only routing with Rhai scripts. Perfect solution."
          avatar={{ from: "green-400", to: "green-600" }}
        />
      </div>
    </SectionContainer>
  )
}
