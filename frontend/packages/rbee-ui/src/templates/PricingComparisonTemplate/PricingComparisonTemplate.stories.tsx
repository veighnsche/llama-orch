import type { Meta, StoryObj } from '@storybook/react'
import { PricingComparisonTemplate } from './PricingComparisonTemplate'

const meta = {
  title: 'Templates/PricingComparisonTemplate',
  component: PricingComparisonTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingComparisonTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnPricingPage: Story = {
  args: {
    title: 'Detailed Feature Comparison',
    subtitle: 'What changes across Home/Lab, Team, and Enterprise.',
    lastUpdated: 'This month',
    legend: {
      includedText: 'Included',
      notAvailableText: 'Not available',
    },
    keyDifferences: [
      'Team adds Web UI + collaboration',
      'Enterprise adds SLA + white-label + services',
      'All plans support unlimited GPUs',
    ],
    columns: [
      {
        key: 'h',
        label: 'Home/Lab',
        subtitle: 'Solo / Homelab',
      },
      {
        key: 't',
        label: 'Team',
        subtitle: 'Small teams',
        badge: 'Best for most teams',
        accent: true,
      },
      {
        key: 'e',
        label: 'Enterprise',
        subtitle: 'Security & SLA',
      },
    ],
    groups: [
      { id: 'core', label: 'Core Platform' },
      { id: 'productivity', label: 'Productivity' },
      { id: 'support', label: 'Support & Services' },
    ],
    rows: [
      {
        feature: 'Number of GPUs',
        group: 'core',
        values: { h: 'Unlimited', t: 'Unlimited', e: 'Unlimited' },
      },
      {
        feature: 'OpenAI-compatible API',
        group: 'core',
        values: { h: true, t: true, e: true },
      },
      {
        feature: 'Multi-GPU orchestration (one or many nodes)',
        group: 'core',
        values: { h: true, t: true, e: true },
      },
      {
        feature: 'Web UI (manage nodes, models, jobs)',
        group: 'productivity',
        values: { h: false, t: true, e: true },
      },
      {
        feature: 'Team collaboration',
        group: 'productivity',
        values: { h: false, t: true, e: true },
      },
      {
        feature: 'Support',
        group: 'support',
        values: {
          h: 'Community',
          t: 'Priority email (business hours)',
          e: 'Dedicated (SLA-backed)',
        },
      },
    ],
    tableCaption: 'Feature availability comparison across Home/Lab, Team, and Enterprise plans.',
    cta: {
      text: 'Ready to get started?',
      buttons: [
        { text: 'Start with Team', href: '/signup' },
        { text: 'Talk to Sales', href: '/contact', variant: 'outline' },
      ],
    },
  },
}
