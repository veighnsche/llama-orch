import type { Meta, StoryObj } from '@storybook/react'
import { Building2, Heart, Scale } from 'lucide-react'
import { IndustryCaseCard } from './IndustryCaseCard'

const meta: Meta<typeof IndustryCaseCard> = {
  title: 'Molecules/IndustryCaseCard',
  component: IndustryCaseCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The IndustryCaseCard molecule displays a regulated industry use case with challenges and solutions in a contrast format. Includes compliance badges and optional link.

## Composition
This molecule is composed of:
- **IconPlate**: Industry icon
- **Industry**: Industry name
- **Segments**: Industry segments description
- **Badges**: Optional compliance badges
- **Summary**: Brief use case summary
- **Challenge Panel**: List of challenges
- **Solution Panel**: List of solutions with rbee
- **Link**: Optional "Learn more" link

## When to Use
- Showcasing industry-specific use cases
- Regulated industry pages
- Compliance-focused content
- Enterprise sales materials

## Used In
- **EnterpriseUseCases**: Displays regulated industry use cases (Financial, Healthcare, Legal, Government)
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    icon: {
      control: false,
      description: 'Lucide icon component (e.g., Building2, Heart, Scale, Shield)',
      table: {
        type: { summary: 'LucideIcon' },
        category: 'Content',
      },
    },
    industry: {
      control: 'text',
      description: 'Industry name',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    segments: {
      control: 'text',
      description: 'Industry segments',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    summary: {
      control: 'text',
      description: 'Use case summary',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    challenges: {
      control: 'object',
      description: 'List of challenges',
      table: {
        type: { summary: 'string[]' },
        category: 'Content',
      },
    },
    solutions: {
      control: 'object',
      description: 'List of solutions',
      table: {
        type: { summary: 'string[]' },
        category: 'Content',
      },
    },
    badges: {
      control: 'object',
      description: 'Optional compliance badges',
      table: {
        type: { summary: 'string[]' },
        category: 'Content',
      },
    },
    href: {
      control: 'text',
      description: 'Optional link to industry page',
      table: {
        type: { summary: 'string' },
        category: 'Behavior',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof IndustryCaseCard>

export const Default: Story = {
  args: {
    icon: Building2,
    industry: 'Financial Services',
    segments: 'Banks, Insurance, FinTech',
    summary: 'Process sensitive financial data while maintaining strict regulatory compliance.',
    challenges: [
      'GDPR and PSD2 compliance required',
      'Customer data must stay in EU',
      'Audit trails for all AI decisions',
    ],
    solutions: [
      'Dutch hosting ensures GDPR compliance',
      'Immutable audit logs for all inferences',
      'Zero data retention policies',
    ],
  },
}

export const WithIcon: Story = {
  args: {
    icon: Heart,
    industry: 'Healthcare',
    segments: 'Hospitals, Clinics, Research',
    summary: 'Analyze patient data and medical records with HIPAA and GDPR compliance.',
    challenges: ['HIPAA and GDPR compliance', 'Patient data privacy', 'Medical device regulations'],
    solutions: ['End-to-end encryption', 'Access control and audit trails', 'Certified infrastructure'],
    badges: ['HIPAA', 'GDPR', 'ISO 27001'],
  },
}

export const WithMetrics: Story = {
  args: {
    icon: Scale,
    industry: 'Legal Services',
    segments: 'Law Firms, Courts, Compliance',
    summary: 'Analyze legal documents and contracts while maintaining attorney-client privilege.',
    challenges: [
      'Attorney-client privilege protection',
      'Document confidentiality',
      'Regulatory compliance',
      'Audit requirements',
    ],
    solutions: [
      'Zero-knowledge architecture',
      'Client-side encryption',
      'Immutable audit trails',
      'Dutch legal framework protection',
    ],
    badges: ['GDPR', 'ISO 27001', 'SOC 2'],
    href: '/industries/legal',
  },
}

export const InUseCasesContext: Story = {
  render: () => (
    <div className="w-full max-w-6xl">
      <div className="mb-4 text-sm text-muted-foreground">Example: IndustryCaseCard in EnterpriseUseCases organism</div>
      <div className="grid gap-6 md:grid-cols-2">
        <IndustryCaseCard
          icon={Building2}
          industry="Financial Services"
          segments="Banks, Insurance, FinTech"
          summary="Process sensitive financial data while maintaining strict regulatory compliance."
          challenges={[
            'GDPR and PSD2 compliance required',
            'Customer data must stay in EU',
            'Audit trails for all AI decisions',
          ]}
          solutions={[
            'Dutch hosting ensures GDPR compliance',
            'Immutable audit logs for all inferences',
            'Zero data retention policies',
          ]}
          badges={['GDPR', 'PSD2', 'SOC 2']}
          href="/industries/financial"
        />
        <IndustryCaseCard
          icon={Heart}
          industry="Healthcare"
          segments="Hospitals, Clinics, Research"
          summary="Analyze patient data and medical records with HIPAA and GDPR compliance."
          challenges={['HIPAA and GDPR compliance', 'Patient data privacy', 'Medical device regulations']}
          solutions={['End-to-end encryption', 'Access control and audit trails', 'Certified infrastructure']}
          badges={['HIPAA', 'GDPR', 'ISO 27001']}
          href="/industries/healthcare"
        />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'IndustryCaseCard as used in the EnterpriseUseCases organism, showing two regulated industries.',
      },
    },
  },
}
