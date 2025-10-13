import { Shield, Lock, FileCheck, Server } from 'lucide-react'
import { SolutionSection } from '@/components/organisms/SolutionSection/SolutionSection'

export function EnterpriseSolution() {
  return (
    <SolutionSection
      id="how-it-works"
      kicker="How rbee Works"
      title="EU-Native AI Infrastructure That Meets Compliance by Design"
      subtitle="rbee provides enterprise-grade AI infrastructure that keeps data sovereign, auditable, and fully under your control. Self-hosted, EU-resident, zero US cloud dependencies."
      features={[
        {
          icon: <Shield className="h-8 w-8" aria-hidden="true" />,
          title: '100% Data Sovereignty',
          body: 'Data never leaves your infrastructure. EU-only deployment. Complete control.',
        },
        {
          icon: <Lock className="h-8 w-8" aria-hidden="true" />,
          title: '7-Year Audit Retention',
          body: 'GDPR-compliant audit logs. Immutable, tamper-evident, legally defensible.',
        },
        {
          icon: <FileCheck className="h-8 w-8" aria-hidden="true" />,
          title: '32 Audit Event Types',
          body: 'Complete visibility. Authentication, data access, compliance events.',
        },
        {
          icon: <Server className="h-8 w-8" aria-hidden="true" />,
          title: 'Zero US Cloud Dependencies',
          body: 'Self-hosted or EU marketplace. No Schrems II concerns. Full compliance.',
        },
      ]}
      steps={[
        {
          title: 'Deploy On-Premises',
          body: 'Install rbee on your EU-based infrastructure. Full air-gap support.',
        },
        {
          title: 'Configure Compliance Policies',
          body: 'Set data residency rules, audit retention, and access controls via Rhai policies.',
        },
        {
          title: 'Enable Audit Logging',
          body: 'Immutable audit trail captures all authentication, data access, and compliance events.',
        },
        {
          title: 'Run Compliant AI',
          body: 'Your models, your data, your infrastructure. Zero external dependencies.',
        },
      ]}
      earnings={{
        title: 'Compliance Metrics',
        rows: [
          {
            model: 'Data Sovereignty',
            meta: 'GDPR Art. 44',
            value: '100%',
            note: 'EU-only',
          },
          {
            model: 'Audit Retention',
            meta: 'GDPR Art. 30',
            value: '7 years',
            note: 'immutable',
          },
          {
            model: 'Security Layers',
            meta: 'Defense-in-depth',
            value: '5 layers',
            note: 'zero-trust',
          },
        ],
        disclaimer: 'rbee is designed to meet GDPR, NIS2, and EU AI Act requirements. Consult your legal team for certification.',
      }}
      ctaPrimary={{
        label: 'Request Demo',
        href: '/enterprise/demo',
      }}
      ctaSecondary={{
        label: 'View Compliance Docs',
        href: '/docs/compliance',
      }}
    />
  )
}
