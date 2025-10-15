import type { Meta, StoryObj } from '@storybook/react'
import { CheckCircle, Lock, Shield } from 'lucide-react'
import { TrustIndicator } from './TrustIndicator'

const meta: Meta<typeof TrustIndicator> = {
  title: 'Molecules/TrustIndicator',
  component: TrustIndicator,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The TrustIndicator molecule displays a trust/security indicator with icon and text.

## Used In
- Security badges
- Trust signals
- Compliance indicators
				`,
      },
    },
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof TrustIndicator>

export const Default: Story = {
  args: {
    icon: Shield,
    text: 'GDPR Compliant',
    variant: 'default',
  },
}

export const WithIcon: Story = {
  args: {
    icon: Lock,
    text: 'End-to-End Encrypted',
    variant: 'primary',
  },
}

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-col gap-4 p-8">
      <TrustIndicator icon={Shield} text="Default variant" variant="default" />
      <TrustIndicator icon={Lock} text="Primary variant" variant="primary" />
      <TrustIndicator icon={CheckCircle} text="Success variant" variant="success" />
    </div>
  ),
}

export const WithTooltip: Story = {
  args: {
    icon: Shield,
    text: 'SOC2 Type II',
    variant: 'success',
  },
}
