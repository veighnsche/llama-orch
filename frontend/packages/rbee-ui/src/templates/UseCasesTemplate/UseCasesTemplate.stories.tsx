import type { Meta, StoryObj } from '@storybook/react'
import { Building, Code, Home as HomeIcon, Laptop, Users, Workflow } from 'lucide-react'
import { UseCasesTemplate } from './UseCasesTemplate'

const meta = {
  title: 'Templates/UseCasesTemplate',
  component: UseCasesTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
  argTypes: {
    columns: {
      control: 'select',
      options: [2, 3],
      description: 'Number of columns in grid',
      table: {
        type: { summary: 'number' },
        defaultValue: { summary: '3' },
      },
    },
  },
} satisfies Meta<typeof UseCasesTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnHomePage: Story = {
  args: {
    items: [
      {
        icon: Laptop,
        title: 'The solo developer',
        scenario: 'Shipping a SaaS with AI features; wants control without vendor lock-in.',
        solution: 'Run rbee on your gaming PC + spare workstation. Llama 70B for coding, SD for assets—local & fast.',
        outcome: '$0/month AI costs. Full control. No rate limits.',
      },
      {
        icon: Users,
        title: 'The small team',
        scenario: '5-person startup burning $500/mo on APIs.',
        solution:
          'Pool 3 workstations + 2 Macs into one rbee cluster. Shared models, faster inference, fewer blockers.',
        outcome: '$6,000+ saved per year. GDPR-friendly by design.',
      },
      {
        icon: HomeIcon,
        title: 'The homelab enthusiast',
        scenario: 'Four GPUs gathering dust.',
        solution: 'Spread workers across your LAN in minutes. Build agents: coder, doc generator, code reviewer.',
        outcome: 'Idle GPUs → productive. Auto-download models, clean shutdowns.',
      },
      {
        icon: Building,
        title: 'The enterprise',
        scenario: '50-dev org. Code cannot leave the premises.',
        solution: 'On-prem rbee with audit trails and policy routing. Rhai-based rules for data residency & access.',
        outcome: 'EU-only compliance. Zero external dependencies.',
      },
      {
        icon: Code,
        title: 'The AI-dependent coder',
        scenario:
          'Building complex codebases with Claude/GPT-4. Fears provider changes, shutdowns, or price hikes.',
        solution:
          'Build your own AI coders with rbee + llama-orch-utils. OpenAI-compatible API runs on YOUR hardware.',
        outcome:
          'Complete independence. Models never change without permission. $0/month forever.',
      },
      {
        icon: Workflow,
        title: 'The agentic AI builder',
        scenario:
          'Needs to build custom AI agents: code generators, doc writers, test creators, code reviewers.',
        solution:
          'Use llama-orch-utils TypeScript library: file ops, LLM invocation, prompt management, response extraction.',
        outcome:
          'Build production AI agents in hours. Full control. No rate limits. Test reproducibility built-in.',
      },
    ],
    columns: 3,
  },
}
