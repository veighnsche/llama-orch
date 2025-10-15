'use client'

import { IndustriesHero } from '@rbee/ui/icons'
import { IndustryCard, type IndustryCardProps, SectionContainer } from '@rbee/ui/molecules'
import { Banknote, Factory, GraduationCap, Heart, Landmark, Scale } from 'lucide-react'

const industries: IndustryCardProps[] = [
  {
    title: 'Financial Services',
    icon: Banknote,
    color: 'primary',
    badge: 'GDPR',
    copy: 'GDPR-ready with audit trails and data residency. Run AI code review and risk analysis without sending financial data to external APIs.',
    anchor: 'finance',
  },
  {
    title: 'Healthcare',
    icon: Heart,
    color: 'chart-2',
    badge: 'HIPAA',
    copy: 'HIPAA-compliant by design. Patient data stays on your network while AI assists with medical coding, documentation, and research.',
    anchor: 'healthcare',
  },
  {
    title: 'Legal',
    icon: Scale,
    color: 'chart-3',
    copy: 'Preserve attorney–client privilege. Perform document/contract analysis and legal research with AI—without client data leaving your environment.',
    anchor: 'legal',
  },
  {
    title: 'Government',
    icon: Landmark,
    color: 'chart-4',
    badge: 'ITAR',
    copy: 'Sovereign, no foreign cloud dependency. Full auditability and policy-enforced routing to meet government security standards.',
    anchor: 'government',
  },
  {
    title: 'Education',
    icon: GraduationCap,
    color: 'chart-2',
    badge: 'FERPA',
    copy: 'Protect student information (FERPA-friendly). AI tutoring, grading assistance, and research tools with zero third-party data sharing.',
    anchor: 'education',
  },
  {
    title: 'Manufacturing',
    icon: Factory,
    color: 'primary',
    copy: 'Safeguard IP and trade secrets. AI-assisted CAD review, quality control, and process optimization—no exposure of proprietary designs.',
    anchor: 'manufacturing',
  },
]

const filters = [
  { label: 'All', anchor: '#architecture' },
  { label: 'Finance', anchor: '#finance' },
  { label: 'Healthcare', anchor: '#healthcare' },
  { label: 'Legal', anchor: '#legal' },
  { label: 'Public Sector', anchor: '#government' },
  { label: 'Education', anchor: '#education' },
  { label: 'Manufacturing', anchor: '#manufacturing' },
]

export function UseCasesIndustry() {
  const handleFilterClick = (anchor: string) => {
    const element = document.querySelector(anchor)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <SectionContainer
      title="Industry-Specific Solutions"
      bgVariant="secondary"
      subtitle="rbee adapts to the unique compliance and security requirements of regulated industries."
    >
      {/* Header block */}
      <div className="max-w-6xl mx-auto mb-8 animate-in fade-in duration-400">
        <p className="text-center text-sm text-muted-foreground mb-6">Regulated sectors · Private-by-design</p>

        {/* Hero banner */}
        <div className="overflow-hidden rounded-lg border/60 mb-8">
          <IndustriesHero
            size="100%"
            className="w-full h-auto"
            aria-label="Visual representation of various industry sectors including healthcare, government, finance, education, manufacturing, and research with AI integration"
          />
        </div>

        {/* Filter pills */}
        <nav
          aria-label="Filter industries"
          className="flex flex-wrap items-center justify-center gap-2 animate-in slide-in-from-top-2 duration-400 delay-75"
        >
          {filters.map((filter) => (
            <button
              key={filter.label}
              onClick={() => handleFilterClick(filter.anchor)}
              className="inline-flex items-center rounded-full border/60 bg-card px-4 py-2 text-sm font-medium text-foreground hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring transition-colors"
            >
              {filter.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Responsive grid: 1 col mobile, 2 cols tablet, 3 cols desktop */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 lg:gap-8 max-w-6xl mx-auto">
        {industries.map((industry, index) => (
          <IndustryCard key={industry.title} {...industry} style={{ animationDelay: `${index * 60}ms` }} />
        ))}
      </div>
    </SectionContainer>
  )
}
