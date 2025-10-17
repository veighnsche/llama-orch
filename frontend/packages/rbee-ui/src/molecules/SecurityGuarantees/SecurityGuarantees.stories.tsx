import type { Meta, StoryObj } from '@storybook/react'
import { SecurityGuarantees } from './SecurityGuarantees'

const meta = {
  title: 'Molecules/Footers/SecurityGuarantees',
  component: SecurityGuarantees,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SecurityGuarantees>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    heading: 'Our Security Guarantees',
    stats: [
      {
        value: '< 15 min',
        label: 'Token expiry time',
        ariaLabel: 'Token expiry time is less than 15 minutes',
      },
      {
        value: '100%',
        label: 'Audit trail coverage',
        ariaLabel: 'One hundred percent audit trail coverage',
      },
      {
        value: '7 years',
        label: 'Log retention (GDPR)',
        ariaLabel: 'Seven years of log retention for GDPR compliance',
      },
    ],
    footnote: 'All security measures are independently audited and certified annually.',
  },
}

export const TwoStats: Story = {
  args: {
    heading: 'Performance Metrics',
    stats: [
      {
        value: '99.9%',
        label: 'Uptime SLA',
      },
      {
        value: '< 100ms',
        label: 'Response time',
      },
    ],
    footnote: 'Measured over the last 12 months.',
  },
}

export const FourStats: Story = {
  args: {
    heading: 'Security by the Numbers',
    stats: [
      {
        value: '256-bit',
        label: 'AES encryption',
      },
      {
        value: '32',
        label: 'Event types tracked',
      },
      {
        value: '0',
        label: 'Security incidents',
      },
      {
        value: 'SOC 2',
        label: 'Type II certified',
      },
    ],
    footnote: 'Independently verified by third-party auditors.',
  },
}
