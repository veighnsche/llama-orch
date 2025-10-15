'use client'

import { UsecasesGridDark } from '@rbee/ui/icons'
import { SectionContainer, UseCaseCard } from '@rbee/ui/molecules'
import { Briefcase, Building, Code, GraduationCap, Home, Laptop, Server, Users } from 'lucide-react'

const useCases: any[] = [
  {
    icon: Laptop,
    color: 'chart-2',
    title: 'The Solo Developer',
    scenario: 'Building a SaaS with AI, wants Claude-level coding without vendor lock-in.',
    solution: 'Run rbee on gaming PC + spare workstation; Llama 70B for code, SD for assets.',
    highlights: ['$0/mo inference', 'Full control', 'No rate limits'],
    anchor: 'developers',
  },
  {
    icon: Users,
    color: 'primary',
    title: 'The Small Team',
    scenario: '5-person startup spending ~$500/mo on AI APIs; needs to cut burn.',
    solution: 'Pool 3 workstations + 2 Macs (8 GPUs) into one rbee cluster.',
    highlights: ['~$6k/yr saved', 'Faster tokens', 'GDPR-friendly'],
    badge: 'Most Popular',
  },
  {
    icon: Home,
    color: 'chart-3',
    title: 'The Homelab Enthusiast',
    scenario: 'Has 4 GPUs collecting dust; wants to build AI agents for personal projects.',
    solution: 'Run rbee across homelab; build custom AI coder, docs generator, code reviewer.',
    highlights: ['Idle hardware → productive', 'Zero ongoing costs', 'Full customization'],
    anchor: 'homelab',
  },
  {
    icon: Building,
    color: 'chart-4',
    title: 'The Enterprise',
    scenario: "50-dev team; code can't leave network due to compliance.",
    solution: 'On-prem rbee with 20 GPUs; custom Rhai routing for data residency.',
    highlights: ['EU-only routing', 'Full audit trail', 'Zero external deps'],
    anchor: 'enterprise',
    badge: 'GDPR',
  },
  {
    icon: Briefcase,
    color: 'primary',
    title: 'The Freelance Developer',
    scenario: "Works on multiple client projects; can't share code with external APIs.",
    solution: 'Run rbee locally; all client code stays on machine; Llama for generation.',
    highlights: ['Client confidentiality', 'Professional AI tools', 'Zero subscriptions'],
  },
  {
    icon: GraduationCap,
    color: 'chart-2',
    title: 'The Research Lab',
    scenario: 'University lab with grant funding; limited budget for cloud services.',
    solution: 'Deploy rbee on lab GPU cluster; use grant for hardware, not subscriptions.',
    highlights: ['Maximize research budget', 'Reproducible experiments', 'No vendor lock-in'],
  },
  {
    icon: Code,
    color: 'chart-3',
    title: 'The Open Source Maintainer',
    scenario: "Maintains popular OSS projects; wants AI for reviews/docs but can't afford enterprise.",
    solution: 'Run rbee on personal hardware; build custom agents for PR reviews, docs, triage.',
    highlights: ['Sustainable AI tooling', 'Community-aligned', 'Zero ongoing costs'],
  },
  {
    icon: Server,
    color: 'chart-4',
    title: 'The GPU Provider',
    scenario: 'Has idle GPU hardware (former mining rig, gaming PC); wants to monetize.',
    solution: 'Join rbee marketplace (M3); set pricing and availability; earn passive income.',
    highlights: ['Passive income stream', 'Help the community', 'Control availability'],
  },
]

const filters = [
  { label: 'All', anchor: '#use-cases' },
  { label: 'Solo', anchor: '#developers' },
  { label: 'Team', anchor: '#use-cases' },
  { label: 'Enterprise', anchor: '#enterprise' },
  { label: 'Research', anchor: '#use-cases' },
]

export function UseCasesPrimary() {
  const handleFilterClick = (anchor: string) => {
    const element = document.querySelector(anchor)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }

  return (
    <SectionContainer title="Real Scenarios. Real Solutions." bgVariant="background">
      {/* Header block with eyebrow */}
      <div className="max-w-6xl mx-auto mb-8 animate-in fade-in duration-500">
        <p className="text-center text-sm text-muted-foreground mb-6">OpenAI-compatible · Your GPUs · Zero API fees</p>

        {/* Hero strip image */}
        <div className="relative overflow-hidden rounded-lg border/60 mb-8">
          <UsecasesGridDark
            size="100%"
            className="w-full h-auto"
            aria-label="Dark themed grid visualization showing various LLM use cases including code generation, documentation writing, and chat interactions"
          />
        </div>

        {/* Filter pills */}
        <nav
          aria-label="Filter use cases"
          className="flex flex-wrap items-center justify-center gap-2 animate-in slide-in-from-top-2 duration-500 delay-100"
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

      {/* Responsive grid: 1 col mobile, 2 cols tablet+ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 lg:gap-8 max-w-6xl mx-auto">
        {useCases.map((useCase, index) => (
          <UseCaseCard key={useCase.title} {...useCase} style={{ animationDelay: `${index * 60}ms` }} />
        ))}
      </div>
    </SectionContainer>
  )
}
