import type { Meta, StoryObj } from '@storybook/react'
import { ListCard } from './ListCard'

const meta = {
  title: 'Molecules/ListCard',
  component: ListCard,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ListCard>

export default meta
type Story = StoryObj<typeof meta>

export const Challenge: Story = {
  args: {
    title: 'Challenge',
    items: [
      'Strict data residency requirements',
      'Complex compliance frameworks',
      'High security standards',
      'Limited vendor options',
    ],
    variant: 'dot',
    color: 'muted',
  },
}

export const Solution: Story = {
  args: {
    title: 'Solution with rbee',
    items: [
      'Dutch data centers with full sovereignty',
      'Pre-certified compliance frameworks',
      'Enterprise-grade security',
      'Transparent operations',
    ],
    variant: 'check',
    color: 'chart-3',
    cardClassName: 'border-chart-3/50 bg-chart-3/10',
    titleClassName: 'text-chart-3',
  },
}

export const Features: Story = {
  args: {
    title: 'Key Features',
    items: ['Real-time processing', 'Scalable infrastructure', 'Advanced analytics', 'Custom integrations'],
    variant: 'arrow',
    color: 'primary',
    cardClassName: 'border-primary/30 bg-primary/5',
    titleClassName: 'text-primary',
  },
}

export const WithPlate: Story = {
  args: {
    title: 'Benefits',
    items: ['Reduced costs', 'Improved performance', 'Better compliance', 'Enhanced security'],
    variant: 'check',
    color: 'chart-3',
    showPlate: true,
  },
}

export const ManyItems: Story = {
  args: {
    title: 'Comprehensive Checklist',
    items: [
      'Infrastructure provisioning',
      'Security hardening',
      'Model deployment',
      'Integration testing',
      'User acceptance testing',
      'Production launch',
      'Monitoring setup',
      'Documentation',
      'Training sessions',
      'Support handoff',
    ],
    variant: 'dot',
    color: 'chart-1',
  },
}
