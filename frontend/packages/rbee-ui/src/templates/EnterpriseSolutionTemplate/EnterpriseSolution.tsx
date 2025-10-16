import { FileCheck, Lock, Server, Shield } from 'lucide-react'
import Image from 'next/image'
import { EnterpriseSolutionTemplate } from './EnterpriseSolutionTemplate'

export function EnterpriseSolution() {
  return (
    <EnterpriseSolutionTemplate
      id="how-it-works"
      kicker="How rbee Works"
      eyebrowIcon={<Shield className="h-4 w-4" aria-hidden="true" />}
      title="EU-Native AI Infrastructure That Meets Compliance by Design"
      subtitle="Enterprise-grade, self-hosted AI that keeps data sovereign, auditable, and under your control—EU resident, zero US cloud dependencies."
      features={[
        {
          icon: <Shield className="h-6 w-6" aria-hidden="true" />,
          title: '100% Data Sovereignty',
          body: 'Data stays on your infrastructure. EU-only deployment. Full control.',
          badge: 'GDPR Art. 44',
        },
        {
          icon: <Lock className="h-6 w-6" aria-hidden="true" />,
          title: '7-Year Audit Retention',
          body: 'Immutable, tamper-evident logs. Legally defensible.',
          badge: 'GDPR Art. 30',
        },
        {
          icon: <FileCheck className="h-6 w-6" aria-hidden="true" />,
          title: '32 Audit Event Types',
          body: 'Auth, data access, policy changes, compliance events.',
        },
        {
          icon: <Server className="h-6 w-6" aria-hidden="true" />,
          title: 'Zero US Cloud Dependencies',
          body: 'Self-hosted or EU marketplace. No Schrems II exposure.',
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
        disclaimer:
          'rbee is designed to meet GDPR, NIS2, and EU AI Act requirements. Consult your legal team for certification.',
      }}
      illustration={
        <Image
          src="/decor/eu-ledger-grid.webp"
          width={1200}
          height={640}
          className="pointer-events-none absolute left-1/2 top-8 -z-10 hidden w-[52rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
          alt="Abstract EU-blue ledger grid with softly glowing checkpoints, implying immutable audit trails and data sovereignty; premium dark UI, subtle amber accents"
          aria-hidden="true"
        />
      }
      ctaPrimary={{
        label: 'Request Demo',
        href: '/enterprise/demo',
      }}
      ctaSecondary={{
        label: 'View Compliance Docs',
        href: '/docs/compliance',
      }}
      ctaCaption="EU data residency guaranteed; earnings/metrics depend on configuration."
    />
  )
}
