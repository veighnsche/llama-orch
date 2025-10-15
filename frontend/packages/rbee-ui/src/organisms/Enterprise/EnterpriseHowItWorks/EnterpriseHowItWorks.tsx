import { StepCard } from '@rbee/ui/molecules'
import { CheckCircle, Rocket, Server, Shield } from 'lucide-react'
import Image from 'next/image'

const deploymentSteps = [
  {
    index: 1,
    icon: <Shield className="h-6 w-6" />,
    title: 'Compliance Assessment',
    intro:
      'We map requirements (GDPR, SOC2, ISO 27001, HIPAA, PCI-DSS), define residency, retention, and security controls.',
    items: [
      'Compliance gap analysis',
      'Data flow mapping',
      'Risk assessment report',
      'Deployment architecture proposal',
    ],
  },
  {
    index: 2,
    icon: <Server className="h-6 w-6" />,
    title: 'On-Premises Deployment',
    intro:
      'Deploy in EU data centers or on your servers. Configure EU-only workers, audit logging, and controls. White-label optional.',
    items: [
      'EU data centers (Frankfurt, Amsterdam, Paris)',
      'On-premises (your servers)',
      'Private cloud (AWS EU, Azure EU, GCP EU)',
      'Hybrid (on-prem + marketplace)',
    ],
  },
  {
    index: 3,
    icon: <CheckCircle className="h-6 w-6" />,
    title: 'Compliance Validation',
    intro:
      'Work with auditors: provide audit-trail access, docs, and architecture reviews. Supports SOC2 Type II, ISO 27001, GDPR.',
    items: [
      'Compliance documentation package',
      'Auditor access to audit logs',
      'Security architecture review',
      'Penetration testing reports',
    ],
  },
  {
    index: 4,
    icon: <Rocket className="h-6 w-6" />,
    title: 'Production Launch',
    intro: 'Go live with enterprise SLAs, 24/7 support, monitoring, and compliance reporting. Scale as you grow.',
    items: [
      '99.9% uptime SLA',
      '24/7 support (1-hour response time)',
      'Dedicated account manager',
      'Quarterly compliance reviews',
    ],
  },
]

export function EnterpriseHowItWorks() {
  return (
    <section
      id="deployment"
      aria-labelledby="deploy-h2"
      className="relative border-b border-border bg-background px-6 py-24"
    >
      {/* Decorative background illustration */}
      <Image
        src="/decor/deployment-flow.webp"
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[48rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt="Abstract EU-blue flow diagram with four checkpoints and connecting lines, suggesting enterprise deployment stages and compliance handoffs"
        aria-hidden="true"
      />

      <div className="relative z-10 mx-auto max-w-7xl">
        {/* Header */}
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 mb-16 text-center duration-500">
          <p className="mb-2 text-sm font-medium text-primary/70">Deployment & Compliance</p>
          <h2 id="deploy-h2" className="mb-4 text-4xl font-bold text-foreground">
            Enterprise Deployment Process
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-foreground/85">
            From consultation to production, we guide every step of your compliance journey.
          </p>
        </div>

        {/* Grid: Steps + Timeline */}
        <div className="grid gap-10 lg:grid-cols-[1fr_360px]">
          {/* Steps Rail */}
          <ol className="animate-in fade-in-50 space-y-8 [animation-delay:calc(var(--i)*80ms)]">
            {deploymentSteps.map((step, idx) => (
              <StepCard
                key={step.index}
                index={step.index}
                icon={step.icon}
                title={step.title}
                intro={step.intro}
                items={step.items}
                isLast={idx === deploymentSteps.length - 1}
              />
            ))}
          </ol>

          {/* Sticky Timeline Panel */}
          <div className="animate-in fade-in-50 slide-in-from-right-2 [animation-delay:200ms] lg:sticky lg:top-24 lg:self-start">
            <div className="rounded-2xl border border-primary/20 bg-primary/5 p-6">
              <h3 className="mb-2 text-xl font-semibold text-foreground">Typical Deployment Timeline</h3>
              <p className="mb-6 text-sm text-muted-foreground">From consultation to production</p>

              {/* Progress bar */}
              <div className="mb-6 h-1 rounded bg-border">
                <div className="h-full w-1/4 rounded bg-primary" aria-hidden="true" />
              </div>

              {/* Week chips */}
              <ol className="space-y-3" role="list">
                <li className="rounded-xl border bg-background px-3 py-2 transition-colors hover:bg-secondary">
                  <div className="text-sm font-semibold text-primary">Week 1-2</div>
                  <div className="text-xs text-muted-foreground">Compliance Assessment</div>
                </li>
                <li className="rounded-xl border bg-background px-3 py-2 transition-colors hover:bg-secondary">
                  <div className="text-sm font-semibold text-primary">Week 3-4</div>
                  <div className="text-xs text-muted-foreground">Deployment & Configuration</div>
                </li>
                <li className="rounded-xl border bg-background px-3 py-2 transition-colors hover:bg-secondary">
                  <div className="text-sm font-semibold text-primary">Week 5-6</div>
                  <div className="text-xs text-muted-foreground">Compliance Validation</div>
                </li>
                <li className="rounded-xl border bg-background px-3 py-2 transition-colors hover:bg-secondary">
                  <div className="text-sm font-semibold text-primary">Week 7</div>
                  <div className="text-xs text-muted-foreground">Production Launch</div>
                </li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
