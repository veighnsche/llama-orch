import type { Meta, StoryObj } from '@storybook/react'
import { HighlightCard } from './HighlightCard'

const meta = {
  title: 'Atoms/HighlightCard',
  component: HighlightCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    color: {
      control: 'select',
      options: ['primary', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
    },
  },
} satisfies Meta<typeof HighlightCard>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default HighlightCard with chart-3 color
 */
export const Default: Story = {
  args: {
    heading: 'Key Features',
    items: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'],
    color: 'chart-3',
  },
}

/**
 * Compliance example with GDPR requirements
 */
export const ComplianceExample: Story = {
  args: {
    heading: 'GDPR Requirements',
    items: [
      'Data residency: EU-only storage',
      'Audit retention: 7 years minimum',
      'Right to erasure: Automated workflows',
      'Breach notification: 72-hour compliance',
    ],
    color: 'chart-3',
  },
}

/**
 * Security example with chart-1 color
 */
export const SecurityExample: Story = {
  args: {
    heading: 'Security Features',
    items: [
      'End-to-end encryption',
      'Zero-trust architecture',
      'Multi-factor authentication',
      'Regular security audits',
    ],
    color: 'chart-1',
  },
}

/**
 * Primary color variant
 */
export const PrimaryColor: Story = {
  args: {
    heading: 'Benefits',
    items: ['Cost savings', 'Improved performance', 'Better compliance', 'Enhanced security'],
    color: 'primary',
  },
}

/**
 * Long list example
 */
export const LongList: Story = {
  args: {
    heading: 'Comprehensive Checklist',
    items: [
      'Data encryption at rest and in transit',
      'Regular security audits and penetration testing',
      'Compliance with GDPR, SOC2, and ISO 27001',
      'Automated backup and disaster recovery',
      'Role-based access control (RBAC)',
      'Audit logging and monitoring',
      'Incident response procedures',
      'Data retention policies',
    ],
    color: 'chart-3',
  },
}

/**
 * White checkmarks on colored background
 */
export const WhiteCheckmarks: Story = {
  args: {
    heading: 'Trust Service Criteria',
    items: ['Security (CC1-CC9)', 'Availability (A1.1-A1.3)', 'Confidentiality (C1.1-C1.2)'],
    color: 'chart-3',
    checkmarkColor: 'white',
  },
}

/**
 * Disabled checkmarks for non-checked items
 */
export const DisabledCheckmarks: Story = {
  args: {
    heading: 'Compliance Endpoints',
    items: [
      'GET /v2/compliance/data-access',
      'POST /v2/compliance/data-export',
      'POST /v2/compliance/data-deletion',
      'GET /v2/compliance/audit-trail',
    ],
    color: 'chart-3',
    disabledCheckmarks: true,
  },
}

/**
 * All color variants
 */
export const AllColors: Story = {
  args: {
    heading: 'Example',
    items: ['Item 1', 'Item 2', 'Item 3'],
  },
  render: () => (
    <div className="grid grid-cols-2 gap-4 max-w-4xl">
      <HighlightCard heading="Primary" items={['Item 1', 'Item 2', 'Item 3']} color="primary" />
      <HighlightCard heading="Chart 1" items={['Item 1', 'Item 2', 'Item 3']} color="chart-1" />
      <HighlightCard heading="Chart 2" items={['Item 1', 'Item 2', 'Item 3']} color="chart-2" />
      <HighlightCard heading="Chart 3" items={['Item 1', 'Item 2', 'Item 3']} color="chart-3" />
      <HighlightCard heading="Chart 4" items={['Item 1', 'Item 2', 'Item 3']} color="chart-4" />
      <HighlightCard heading="Chart 5" items={['Item 1', 'Item 2', 'Item 3']} color="chart-5" />
    </div>
  ),
}
