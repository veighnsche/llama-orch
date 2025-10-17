import type { Meta, StoryObj } from '@storybook/react'
import { AuditReadinessCTA } from './AuditReadinessCTA'

const meta = {
  title: 'Molecules/AuditReadinessCTA',
  component: AuditReadinessCTA,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof AuditReadinessCTA>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    heading: 'Audit-Ready from Day One',
    description:
      'Our compliance documentation pack includes pre-filled templates, audit trails, and evidence bundles that satisfy ISO 27001, SOC 2, and GDPR requirements.',
    note: 'Available with Enterprise tier or as a standalone compliance pack.',
    noteAriaLabel: 'Compliance pack availability information',
    buttons: [
      {
        text: 'Download Compliance Pack',
        href: '/compliance-pack',
        variant: 'default',
        ariaDescribedby: 'compliance-pack-note',
      },
      {
        text: 'Schedule Audit Walkthrough',
        href: '/contact?topic=audit',
        variant: 'outline',
      },
    ],
    footnote: 'All documentation templates are reviewed by certified auditors and updated quarterly.',
  },
}

export const SingleButton: Story = {
  args: {
    heading: 'Get Started with Compliance',
    description: 'Download our comprehensive compliance starter pack today.',
    note: 'Free for all users.',
    noteAriaLabel: 'Free compliance pack information',
    buttons: [
      {
        text: 'Download Now',
        href: '/download',
        variant: 'default',
      },
    ],
    footnote: 'No credit card required.',
  },
}

export const ThreeButtons: Story = {
  args: {
    heading: 'Multiple Compliance Options',
    description: 'Choose the compliance path that works best for your organization.',
    note: 'All options include full documentation and support.',
    noteAriaLabel: 'Compliance options information',
    buttons: [
      {
        text: 'ISO 27001',
        href: '/iso-27001',
        variant: 'default',
      },
      {
        text: 'SOC 2',
        href: '/soc-2',
        variant: 'outline',
      },
      {
        text: 'GDPR',
        href: '/gdpr',
        variant: 'outline',
      },
    ],
    footnote: 'Can be combined for comprehensive coverage.',
  },
}
