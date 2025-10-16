import { Button } from '@rbee/ui/atoms/Button'
import { IndustryCaseCard } from '@rbee/ui/organisms'
import { Building2, Heart, Scale, Shield } from 'lucide-react'
import Image from 'next/image'
import Link from 'next/link'

const industryCases = [
  {
    icon: <Building2 className="size-6" />,
    industry: 'Financial Services',
    segments: 'Banks, Insurance, FinTech',
    badges: ['PCI-DSS', 'GDPR', 'SOC2'],
    summary: 'EU bank needed internal code-gen but PCI-DSS/GDPR blocked external AI.',
    challenges: [
      'No external APIs (PCI-DSS)',
      'Complete audit trail (SOC2)',
      'EU data residency (GDPR)',
      '7-year retention',
    ],
    solutions: [
      'On-prem (EU data center)',
      'Immutable audit logs (7-year)',
      'Zero external dependencies',
      'SOC2 Type II ready',
    ],
    href: '/industries/finance',
  },
  {
    icon: <Heart className="size-6" />,
    industry: 'Healthcare',
    segments: 'Hospitals, MedTech, Pharma',
    badges: ['HIPAA', 'GDPR Art. 9'],
    summary: 'AI-assisted patient tooling with HIPAA + GDPR Article 9 constraints.',
    challenges: ['HIPAA/PHI protection', 'GDPR Art. 9 (health data)', 'No US clouds', 'Breach notifications'],
    solutions: [
      'Self-hosted (hospital DC)',
      'EU-only deployment',
      'Full audit trail (breach detection)',
      'HIPAA-aligned architecture',
    ],
    href: '/industries/healthcare',
  },
  {
    icon: <Scale className="size-6" />,
    industry: 'Legal Services',
    segments: 'Law Firms, LegalTech',
    badges: ['GDPR', 'Legal Hold'],
    summary: 'Document analysis without risking privilege.',
    challenges: ['Attorney-client privilege', 'No external uploads', 'Legal-hold audit trail', 'EU residency'],
    solutions: ['On-prem (firm servers)', 'Zero data transfer', 'Immutable legal-hold logs', 'Full confidentiality'],
    href: '/industries/legal',
  },
  {
    icon: <Shield className="size-6" />,
    industry: 'Government',
    segments: 'Public Sector, Defense',
    badges: ['ISO 27001', 'Sovereignty'],
    summary: 'Citizen services with strict sovereignty + security controls.',
    challenges: ['Data sovereignty', 'No foreign clouds', 'Transparent audit trail', 'ISO 27001 required'],
    solutions: ['Gov DC deployment', 'EU-only infra', 'ISO 27001 aligned', 'Complete sovereignty'],
    href: '/industries/government',
  },
]

export function EnterpriseUseCases() {
  return (
    <section aria-labelledby="industries-h2" className="relative border-b border-border bg-background px-6 py-24">
      {/* Decorative background illustration */}
      <Image
        src="/decor/sector-grid.webp"
        width={1200}
        height={640}
        className="pointer-events-none absolute left-1/2 top-6 -z-10 hidden w-[50rem] -translate-x-1/2 opacity-15 blur-[0.5px] md:block"
        alt="Abstract EU-blue grid of industry tiles—finance, healthcare, legal, government—with soft amber accents; premium dark UI, compliance theme"
        aria-hidden="true"
      />

      <div className="relative z-10 mx-auto max-w-7xl">
        {/* Header */}
        <div className="animate-in fade-in-50 slide-in-from-bottom-2 mb-16 text-center duration-500">
          <p className="mb-2 text-sm font-medium text-primary/70">Industry Playbooks</p>
          <h2 id="industries-h2" className="mb-4 text-4xl font-bold text-foreground">
            Built for Regulated Industries
          </h2>
          <p className="mx-auto max-w-3xl text-balance text-xl text-foreground/85">
            Organizations in high-compliance sectors run rbee on EU-resident infrastructure—no foreign clouds,
            audit-ready by design.
          </p>
        </div>

        {/* Industry Grid */}
        <div className="animate-in fade-in-50 mb-12 grid gap-8 [animation-delay:120ms] md:grid-cols-2">
          {industryCases.map((industryCase) => (
            <IndustryCaseCard
              key={industryCase.industry}
              icon={industryCase.icon}
              industry={industryCase.industry}
              segments={industryCase.segments}
              badges={industryCase.badges}
              summary={industryCase.summary}
              challenges={industryCase.challenges}
              solutions={industryCase.solutions}
              href={industryCase.href}
            />
          ))}
        </div>

        {/* CTA Rail */}
        <div className="animate-in fade-in-50 rounded-2xl border border-primary/20 bg-primary/5 p-6 text-center [animation-delay:200ms]">
          <p className="mb-6 text-lg font-semibold text-foreground">See how rbee fits your sector.</p>
          <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
            <Button size="lg" asChild className="transition-transform active:scale-[0.98]">
              <Link href="/contact/industry-brief">Request Industry Brief</Link>
            </Button>
            <Button size="lg" variant="outline" asChild className="transition-transform active:scale-[0.98]">
              <Link href="/contact/solutions">Talk to a Solutions Architect</Link>
            </Button>
          </div>
          <div className="mt-4 flex flex-wrap justify-center gap-3 text-sm text-muted-foreground">
            <Link href="/industries/finance" className="hover:text-primary hover:underline">
              Finance
            </Link>
            <span>•</span>
            <Link href="/industries/healthcare" className="hover:text-primary hover:underline">
              Healthcare
            </Link>
            <span>•</span>
            <Link href="/industries/legal" className="hover:text-primary hover:underline">
              Legal
            </Link>
            <span>•</span>
            <Link href="/industries/government" className="hover:text-primary hover:underline">
              Government
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}
