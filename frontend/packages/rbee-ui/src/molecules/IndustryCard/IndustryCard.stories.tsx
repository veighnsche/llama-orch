import type { Meta, StoryObj } from '@storybook/react'
import { Briefcase, Building2, GraduationCap, ShoppingCart } from 'lucide-react'
import { IndustryCard } from './IndustryCard'

const meta: Meta<typeof IndustryCard> = {
  title: 'Molecules/IndustryCard',
  component: IndustryCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The IndustryCard molecule displays an industry use case with icon, title, description, and optional badge. Features hover effects and scroll-to-anchor support.

## Composition
This molecule is composed of:
- **IconBox**: Industry icon with color
- **Badge**: Optional badge (e.g., "Popular", "New")
- **Title**: Industry name
- **Copy**: Description text

## When to Use
- Industry overview pages
- Use case listings
- Sector-specific content
- Market segment displays

## Used In
- **UseCasesIndustry**: Displays industry cards with scroll-to-anchor navigation
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    title: {
      control: 'text',
      description: 'Industry title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    copy: {
      control: 'text',
      description: 'Industry description',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    icon: {
      control: false,
      description: 'Lucide icon component',
      table: {
        type: { summary: 'LucideIcon' },
        category: 'Content',
      },
    },
    color: {
      control: 'select',
      options: ['primary', 'chart-2', 'chart-3', 'chart-4'],
      description: 'Icon color',
      table: {
        type: { summary: "'primary' | 'chart-2' | 'chart-3' | 'chart-4'" },
        category: 'Appearance',
      },
    },
    badge: {
      control: 'text',
      description: 'Optional badge text',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    anchor: {
      control: 'text',
      description: 'Optional anchor ID for scroll-to navigation',
      table: {
        type: { summary: 'string' },
        category: 'Behavior',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof IndustryCard>

export const Default: Story = {
  args: {
    title: 'Financial Services',
    copy: 'Banks, insurance companies, and fintech startups use rbee for secure, compliant AI processing of sensitive financial data.',
    icon: <Building2 className="size-6" />,
    color: 'primary',
  },
}

export const WithIcon: Story = {
  args: {
    title: 'Education',
    copy: 'Universities and online learning platforms leverage rbee for personalized learning experiences while protecting student data.',
    icon: <GraduationCap className="size-6" />,
    color: 'chart-2',
    badge: 'Popular',
  },
}

export const WithExamples: Story = {
  args: {
    title: 'Professional Services',
    copy: 'Consulting firms, law offices, and accounting practices use rbee to analyze documents and generate insights while maintaining client confidentiality.',
    icon: <Briefcase className="size-6" />,
    color: 'chart-3',
    anchor: 'professional-services',
  },
}

export const InIndustryContext: Story = {
  render: () => (
    <div className="w-full max-w-6xl">
      <div className="mb-4 text-sm text-muted-foreground">Example: IndustryCard in UseCasesIndustry organism</div>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <IndustryCard
          title="Financial Services"
          copy="Banks, insurance companies, and fintech startups use rbee for secure, compliant AI processing."
          icon={<Building2 className="size-6" />}
          color="primary"
          badge="GDPR Ready"
          anchor="financial"
        />
        <IndustryCard
          title="Education"
          copy="Universities and online learning platforms leverage rbee for personalized learning experiences."
          icon={<GraduationCap className="size-6" />}
          color="chart-2"
          badge="Popular"
          anchor="education"
        />
        <IndustryCard
          title="E-commerce"
          copy="Online retailers use rbee for product recommendations, customer support, and inventory optimization."
          icon={<ShoppingCart className="size-6" />}
          color="chart-3"
          anchor="ecommerce"
        />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'IndustryCard as used in the UseCasesIndustry organism, showing multiple industries with scroll anchors.',
      },
    },
  },
}
