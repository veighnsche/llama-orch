import { Badge } from '@rbee/ui/atoms/Badge'
import { Button } from '@rbee/ui/atoms/Button'
import { Card } from '@rbee/ui/atoms/Card'
import { GitHubIcon, RbeeArch } from '@rbee/ui/icons'
import type { ArchitectureHighlight, TechItem } from '@rbee/ui/molecules'
import { ArchitectureHighlights, CoverageProgressBar, SectionContainer, TechnologyStack } from '@rbee/ui/molecules'
import { Terminal } from 'lucide-react'
import Link from 'next/link'

const techStack: TechItem[] = [
  {
    name: 'Rust',
    description: 'Performance + memory safety.',
    ariaLabel: 'Tech: Rust',
  },
  {
    name: 'Candle ML',
    description: 'Rust-native inference.',
    ariaLabel: 'Tech: Candle ML',
  },
  {
    name: 'Rhai Scripting',
    description: 'Embedded, sandboxed policies.',
    ariaLabel: 'Tech: Rhai Scripting',
  },
  {
    name: 'SQLite',
    description: 'Embedded, zero-ops DB.',
    ariaLabel: 'Tech: SQLite',
  },
  {
    name: 'Axum + Vue.js',
    description: 'Async backend + modern UI.',
    ariaLabel: 'Tech: Axum + Vue.js',
  },
]

const architectureHighlights: ArchitectureHighlight[] = [
  {
    title: 'BDD-Driven Development',
    details: ['42/62 scenarios passing (68% complete)', 'Live CI coverage'],
  },
  {
    title: 'Cascading Shutdown Guarantee',
    details: ['No orphaned processes. Clean VRAM lifecycle.'],
  },
  {
    title: 'Process Isolation',
    details: ['Worker-level sandboxes. Zero cross-leak.'],
  },
  {
    title: 'Protocol-Aware Orchestration',
    details: ['SSE, JSON, binary protocols.'],
  },
  {
    title: 'Smart/Dumb Separation',
    details: ['Central brain, distributed execution.'],
  },
]

export function TechnicalSection() {
  return (
    <SectionContainer
      title="Built by AI Engineers, for Engineers building with AI"
      description="Rust-native orchestrator with process isolation, protocol awareness, and policy routing via Rhai."
      headingId="tech-title"
      align="center"
    >
      <div className="grid grid-cols-12 gap-6 lg:gap-10 max-w-6xl mx-auto motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-500">
        {/* Left Column: Architecture Highlights + Diagram */}
        <div className="col-span-12 lg:col-span-6 space-y-6">
          {/* Architecture Highlights */}
          <ArchitectureHighlights highlights={architectureHighlights} />

          {/* BDD Coverage Progress Bar */}
          <CoverageProgressBar label="BDD Coverage" passing={42} total={62} className="mt-6" />

          {/* Architecture Diagram (Desktop Only) */}
          <RbeeArch
            className="hidden md:block rounded-2xl ring-1 ring-border/60 shadow-sm"
            aria-label="rbee architecture diagram showing orchestrator, policy engine, and worker pools"
          />
        </div>

        {/* Right Column: Technology Stack (Sticky on Large Screens) */}
        <div className="col-span-12 lg:col-span-6 lg:sticky lg:top-20">
          <TechnologyStack
            technologies={techStack}
            githubUrl="https://github.com/yourusername/rbee"
            license="MIT License"
            architectureUrl="/docs/architecture"
          />
        </div>
      </div>
    </SectionContainer>
  )
}
