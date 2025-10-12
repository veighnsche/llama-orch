import { Code2, Server, Shield } from "lucide-react"
import { AudienceCard } from '@/components/molecules'

export function AudienceSelector() {
  return (
    <section className="relative bg-background py-24 sm:py-32">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/20 to-transparent" />

      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto mb-16 max-w-3xl text-center">
          <p className="mb-4 font-sans text-sm font-medium uppercase tracking-wider text-primary">Choose Your Path</p>

          <h2 className="mb-6 font-sans text-3xl font-semibold tracking-tight text-foreground sm:text-4xl lg:text-5xl">
            Where do you want to start?
          </h2>
          <p className="font-sans text-lg leading-relaxed text-muted-foreground">
            rbee adapts to your needs. Whether you're building, providing, or securing—we have a path designed for you.
          </p>
        </div>

        <div className="mx-auto grid max-w-6xl gap-6 lg:grid-cols-3 lg:gap-8">
          <AudienceCard
            icon={Code2}
            category="For Developers"
            title="Build on Your Hardware"
            description="Use your homelab GPUs to power AI coding tools. OpenAI-compatible API works with Zed, Cursor, and any tool you already use."
            features={[
              "Zero API costs, unlimited usage",
              "Complete privacy, no data leaves your network",
              "Build custom AI agents with TypeScript"
            ]}
            href="/developers"
            ctaText="Explore Developer Path"
            color="chart-2"
          />

          <AudienceCard
            icon={Server}
            category="For GPU Owners"
            title="Monetize Your Hardware"
            description="Turn idle GPUs into revenue. Join the rbee marketplace and earn from your gaming PC, workstation, or server farm."
            features={[
              "Set your own pricing and availability",
              "Secure platform with audit trails",
              "Passive income from existing hardware"
            ]}
            href="/gpu-providers"
            ctaText="Become a Provider"
            color="chart-3"
          />

          <AudienceCard
            icon={Shield}
            category="For Enterprise"
            title="Compliance & Security"
            description="Deploy AI infrastructure with EU compliance, comprehensive audit trails, and enterprise-grade security built-in from day one."
            features={[
              "GDPR-compliant with 7-year audit retention",
              "SOC2 and ISO 27001 aligned",
              "On-premises or private cloud deployment"
            ]}
            href="/enterprise"
            ctaText="Enterprise Solutions"
            color="primary"
          />
        </div>

        <div className="mx-auto mt-12 max-w-2xl text-center">
          <p className="font-sans text-sm leading-relaxed text-muted-foreground">
            Not sure which path fits? Keep scrolling to explore the full platform.
          </p>
        </div>
      </div>
    </section>
  )
}
