import type { Meta, StoryObj } from '@storybook/react'
import { HomeHero } from './HomeHero'
import type { HomeHeroProps } from './HomeHero'

const meta = {
  title: 'Templates/HomeHero',
  component: HomeHero,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof HomeHero>

export default meta
type Story = StoryObj<typeof meta>

const defaultProps: HomeHeroProps = {
  badgeText: '100% Open Source • GPL-3.0-or-later',
  headlinePrefix: 'Run LLMs on',
  headlineHighlight: 'Your Hardware',
  subcopy: 'Self-hosted AI inference orchestrator. Deploy any model on your GPUs with zero vendor lock-in.',
  bullets: [
    { title: 'Deploy any open model', variant: 'check', color: 'chart-3' },
    { title: 'Keep data private', variant: 'check', color: 'chart-3' },
    { title: 'Zero ongoing costs', variant: 'check', color: 'chart-3' },
  ],
  primaryCTA: {
    label: 'Get Started',
    href: '/docs',
    showIcon: true,
  },
  secondaryCTA: {
    label: 'View on GitHub',
    href: 'https://github.com',
    variant: 'outline',
  },
  trustBadges: [
    { type: 'github', label: '2.5k+ stars' },
    { type: 'api', label: 'OpenAI Compatible' },
    { type: 'cost', label: '$0/month hosting' },
  ],
  terminalTitle: 'llama-orch',
  terminalCommand: 'llorch run llama-3.1-8b-instruct',
  terminalOutput: {
    loading: 'Loading model weights...',
    ready: 'Model ready on GPU pool "homelab"',
    prompt: 'Explain quantum computing',
    generating: 'Quantum computing harnesses quantum mechanical phenomena...',
  },
  gpuPoolLabel: 'GPU Pool: homelab',
  gpuProgress: [
    { label: 'RTX 4090 (24GB)', percentage: 65 },
    { label: 'RTX 3090 (24GB)', percentage: 45 },
  ],
  costLabel: 'Cost (last hour)',
  costValue: '$0.00',
  floatingKPI: {
    gpuPool: { label: 'Active GPUs', value: '2/4' },
    cost: { label: 'Cost/hour', value: '$0.00' },
    latency: { label: 'Latency', value: '45ms' },
  },
}

export const OnHomeHero: Story = {
  args: defaultProps,
}
