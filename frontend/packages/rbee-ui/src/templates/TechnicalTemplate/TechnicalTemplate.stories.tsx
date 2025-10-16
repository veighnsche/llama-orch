import type { Meta, StoryObj } from '@storybook/react'
import { RbeeArch } from '@rbee/ui/icons'
import { TechnicalTemplate } from './TechnicalTemplate'

const meta = {
  title: 'Templates/TechnicalTemplate',
  component: TechnicalTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof TechnicalTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: {
    architectureHighlights: [
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
    ],
    coverageProgress: {
      label: 'BDD Coverage',
      passing: 42,
      total: 62,
    },
    architectureDiagram: {
      component: RbeeArch,
      ariaLabel: 'rbee architecture diagram showing orchestrator, policy engine, and worker pools',
    },
    techStack: [
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
    ],
    stackLinks: {
      githubUrl: 'https://github.com/veighnsche/llama-orch',
      license: 'GPL-3.0-or-later',
      architectureUrl: '/docs/architecture',
    },
  },
}
