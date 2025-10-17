import type { Meta, StoryObj } from '@storybook/react'
import CompliancePage from './CompliancePage'

const meta = {
  title: 'Pages/CompliancePage',
  component: CompliancePage,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Compliance page targeting regulated industries. Showcases GDPR compliance, audit trails, data sovereignty, and regulatory compliance features.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CompliancePage>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Complete Compliance page with all sections:
 * - Hero with compliance overview
 * - Regulatory frameworks (GDPR, HIPAA, SOC 2)
 * - Audit trail features
 * - Data sovereignty and privacy
 * - Compliance use cases
 * - Security certifications
 * - FAQ section
 * - Final CTA
 */
export const Default: Story = {}
