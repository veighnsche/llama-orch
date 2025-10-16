import type { Meta, StoryObj } from '@storybook/react'
import { Shield } from 'lucide-react'
import { SecurityCrateCard } from './SecurityCrateCard'

const meta: Meta<typeof SecurityCrateCard> = {
  title: 'Molecules/SecurityCrateCard',
  component: SecurityCrateCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The SecurityCrateCard molecule displays security features with icon, title, and feature list.

## Used In
- Security page
- Compliance sections
				`,
      },
    },
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof SecurityCrateCard>

export const Default: Story = {
  args: {
    icon: <Shield className="size-6" />,
    title: 'Zero-Trust Auth',
    subtitle: 'Authentication',
    description: 'Every request requires valid credentials. No implicit trust.',
    features: ['mTLS certificates', 'API key rotation', 'Role-based access'],
  },
}

export const WithIcon: Story = {
  args: {
    icon: <Shield className="size-6" />,
    title: 'Audit Trails',
    subtitle: 'Compliance',
    description: 'Immutable logs for all operations.',
    features: ['Tamper-proof logs', 'Real-time monitoring', 'Export capabilities'],
  },
}

export const WithDetails: Story = {
  args: {
    icon: <Shield className="size-6" />,
    title: 'Data Encryption',
    subtitle: 'Security',
    description: 'End-to-end encryption for all data.',
    features: ['AES-256 encryption', 'TLS 1.3', 'Key management'],
  },
}

export const Highlighted: Story = {
  args: {
    icon: <Shield className="size-6" />,
    title: 'GDPR Compliance',
    subtitle: 'Regulatory',
    description: 'Full GDPR compliance out of the box.',
    features: ['Data residency', 'Right to erasure', 'Data portability'],
  },
}
