import type { Meta, StoryObj } from '@storybook/react'
import { UsecasesGridDark } from '@rbee/ui/icons'
import { Briefcase, Building, Code, GraduationCap, Home, Laptop, Server, Users } from 'lucide-react'
import { UseCasesPrimaryTemplate } from './UseCasesPrimaryTemplate'

const meta = {
  title: 'Templates/UseCasesPrimaryTemplate',
  component: UseCasesPrimaryTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof UseCasesPrimaryTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnUseCasesPage: Story = {
  args: {
    eyebrow: 'OpenAI-compatible • Your GPUs • Zero API Fees',
    heroImage: <UsecasesGridDark size="100%" className="w-full h-auto" aria-label="Dark grid of LLM use cases: code, docs, chat" />,
    heroImageAriaLabel: 'Dark grid of LLM use cases: code, docs, chat',
    filters: [
      { label: 'All', anchor: '#use-cases' },
      { label: 'Solo', anchor: '#developers' },
      { label: 'Team', anchor: '#use-cases' },
      { label: 'Enterprise', anchor: '#enterprise' },
      { label: 'Research', anchor: '#use-cases' },
    ],
    useCases: [
      {
        icon: Laptop,
        color: 'chart-2',
        title: 'Solo Developer',
        scenario:
          'Building a SaaS with AI assistance—needs Claude-level coding without creating a dependency on external providers. Worried about model changes, pricing shifts, or service shutdowns making the codebase unmaintainable.',
        solution:
          'Run rbee on gaming PC + spare workstation. Use Llama 70B for code generation, Stable Diffusion for assets. OpenAI-compatible API works with existing tools like Zed IDE.',
        outcome:
          'Complete independence. Models never change without permission. Code never leaves your network. Build AI coders that run forever on your hardware.',
        highlights: ['$0 inference', 'Private by default', 'No rate limits'],
        anchor: 'developers',
      },
      {
        icon: Users,
        color: 'primary',
        title: 'Small Team Startup',
        scenario:
          "5-person team burning ~$500/mo on AI APIs for code generation, docs, and support. Need to cut costs without sacrificing speed. Can't afford enterprise pricing but need professional-grade tools.",
        solution:
          'Pool 3 workstations + 2 Macs (8 GPUs total) into one rbee cluster. Multi-node orchestration distributes workload. SSH-based control plane keeps it simple.',
        outcome:
          'Save ~$6k/year while getting faster inference. Team owns the infrastructure. Scale by adding more hardware, not paying higher tiers.',
        highlights: ['Save ~$6k/yr', 'Faster tokens', 'GDPR-friendly'],
        badge: 'Most Popular',
      },
      {
        icon: Home,
        color: 'chart-3',
        title: 'Homelab Builder',
        scenario:
          'Four idle GPUs sitting unused—wants to build personal AI agents for coding, documentation, and automation. Tired of paying subscriptions for tools that could run locally.',
        solution:
          'Federate homelab hardware with rbee. Spin up specialized agents: AI coder for projects, docs generator for wikis, code reviewer for PRs. Use Rhai scripts to route tasks intelligently.',
        outcome:
          'Idle hardware becomes productive infrastructure. Build custom AI workflows without ongoing costs. Full control over models and routing logic.',
        highlights: ['Idle → productive', 'Zero ongoing fees', 'Full customization'],
        anchor: 'homelab',
      },
      {
        icon: Building,
        color: 'chart-4',
        title: 'Enterprise',
        scenario:
          '50-developer organization where code cannot leave the network due to compliance requirements. Need AI-assisted development but cloud APIs violate data residency policies. GDPR and audit trails are mandatory.',
        solution:
          'Deploy on-prem rbee with 20 GPUs. Rhai routing scripts enforce EU-only workers. Built-in audit logging tracks every API call. GDPR endpoints handle data export and deletion requests.',
        outcome:
          'Developers get AI assistance without compliance risk. Full audit trail for regulators. Data never leaves your infrastructure. EU-only routing guaranteed.',
        highlights: ['EU-only routing', 'Full audit trail', 'No external deps'],
        anchor: 'enterprise',
        badge: 'GDPR',
      },
      {
        icon: Briefcase,
        color: 'primary',
        title: 'Freelance Developer',
        scenario:
          'Works on multiple client projects under strict NDAs. Source code cannot touch external APIs or cloud services. Need AI coding assistance but client confidentiality is non-negotiable.',
        solution:
          'Run rbee entirely locally. All client code stays on the machine. Use Llama models for generation, refactoring, and documentation. OpenAI-compatible API works with favorite tools.',
        outcome:
          'Professional AI tools without breaking NDAs. Client code never leaves your laptop. No subscriptions, no usage tracking, complete confidentiality.',
        highlights: ['Client confidentiality', 'Pro-grade tools', 'Zero subscriptions'],
      },
      {
        icon: GraduationCap,
        color: 'chart-2',
        title: 'Research Lab',
        scenario:
          'Grant-funded university lab with tight cloud budget. Need reproducible experiments for papers. Cloud inference costs eat into research funding. Reproducibility requires deterministic outputs.',
        solution:
          'Deploy rbee on the lab GPU cluster. Spend grants on hardware (one-time cost) instead of cloud usage (recurring). Proof bundle system captures seeds and transcripts for reproducible runs.',
        outcome:
          'Maximize research budget by eliminating cloud fees. Reproducible experiments with deterministic testing. Own the infrastructure, control the models.',
        highlights: ['Maximize budget', 'Reproducible runs', 'No vendor lock-in'],
      },
      {
        icon: Code,
        color: 'chart-3',
        title: 'Open Source Maintainer',
        scenario:
          "Maintains popular OSS projects. Needs AI help with PR reviews, documentation generation, and issue triage. Can't afford enterprise API pricing. Want tools that align with open source values.",
        solution:
          'Run rbee on personal hardware. Build custom agents for PR review, docs generation, and issue triage. Use community-shared Rhai routing scripts. GPL license aligns with OSS philosophy.',
        outcome:
          'Sustainable AI tooling for OSS work. Help contributors without burning out. Community-aligned infrastructure that respects open source values.',
        highlights: ['Sustainable tooling', 'Community-aligned', 'No ongoing fees'],
      },
      {
        icon: Server,
        color: 'chart-4',
        title: 'GPU Provider',
        scenario:
          'Idle GPUs from former mining rigs or gaming PCs sitting unused. Want predictable monetization without complex setup. Interested in helping the community while earning passive income.',
        solution:
          'Join the rbee marketplace (M3 milestone). Set your own pricing and availability. Platform handles routing, billing, and SLA monitoring. Task-based pricing means you only earn when GPUs are used.',
        outcome:
          'Turn idle hardware into passive income. Help developers access affordable compute. Control your supply and pricing. Join a community-driven marketplace.',
        highlights: ['Passive income', 'Help the community', 'You control supply'],
      },
    ],
  },
}
