import { Shield, Users, Wrench, Globe } from 'lucide-react'
import { FeatureCard } from '@/components/molecules/FeatureCard/FeatureCard'

const FEATURES = [
  {
    icon: <Shield />,
    title: 'Enterprise SLAs',
    intro: '99.9% uptime with 24/7 support and 1-hour response. Dedicated manager and quarterly reviews.',
    bullets: [
      '99.9% SLA',
      '24/7 support (1-hour)',
      'Dedicated account manager',
      'Quarterly reviews',
    ],
  },
  {
    icon: <Users />,
    title: 'White-Label Option',
    intro: 'Run rbee as your brand—custom domain, UI, and endpoints.',
    bullets: [
      'Custom branding/logo',
      'Custom domain (ai.yourcompany.com)',
      'UI customization',
      'API endpoint customization',
    ],
  },
  {
    icon: <Wrench />,
    title: 'Professional Services',
    intro: 'Deployment, integration, optimization, and training from our team.',
    bullets: [
      'Deployment consulting',
      'Integration support',
      'Custom development',
      'Team training',
    ],
  },
  {
    icon: <Globe />,
    title: 'Multi-Region Support',
    intro: 'EU multi-region for redundancy and compliance: failover + load balancing.',
    bullets: [
      'EU multi-region',
      'Automatic failover',
      'Load balancing',
      'Geo-redundancy',
    ],
  },
]

export function EnterpriseFeatures() {
  return (
    <section
      aria-labelledby="enterprise-features-h2"
      className="relative border-b border-border bg-background px-6 py-24 overflow-hidden"
    >
      {/* Decorative Gradient */}
      <div
        className="pointer-events-none absolute inset-0 bg-radial-glow"
        aria-hidden="true"
      />

      <div className="relative mx-auto max-w-7xl">
        {/* Header Block */}
        <div className="mb-12 text-center animate-in fade-in-50 slide-in-from-bottom-2">
          <p className="mb-2 text-sm font-semibold uppercase tracking-wide text-primary">Enterprise Capabilities</p>
          <h2 id="enterprise-features-h2" className="mb-4 text-4xl font-bold text-foreground">
            Enterprise Features
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-muted-foreground">
            Everything you need for compliant, resilient, EU-resident AI infrastructure.
          </p>
        </div>

        {/* Feature Grid */}
        <div className="grid gap-8 md:grid-cols-2 animate-in fade-in-50" style={{ animationDelay: '120ms' }}>
          {FEATURES.map((feature, index) => (
            <FeatureCard key={index} {...feature} />
          ))}
        </div>

        {/* Outcomes Band */}
        <div
          className="mt-10 rounded-2xl border border-primary/20 bg-primary/5 p-6 md:p-8 animate-in fade-in-50"
          style={{ animationDelay: '200ms' }}
        >
          <h3 className="mb-6 text-lg font-semibold text-foreground">What you get</h3>
          <div className="grid gap-6 sm:grid-cols-3">
            <div className="text-center">
              <div className="mb-1 text-3xl font-bold text-foreground">99.9%</div>
              <div className="text-sm text-muted-foreground">Uptime SLA</div>
            </div>
            <div className="text-center">
              <div className="mb-1 text-3xl font-bold text-foreground">&lt; 1 hr</div>
              <div className="text-sm text-muted-foreground">Support response</div>
            </div>
            <div className="text-center">
              <div className="mb-1 text-3xl font-bold text-foreground">EU-only</div>
              <div className="text-sm text-muted-foreground">Data residency</div>
            </div>
          </div>
          <div className="mt-6 text-center">
            <a
              href="#compliance"
              className="inline-flex items-center gap-1 text-sm text-primary hover:underline focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none rounded"
            >
              See compliance details →
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
