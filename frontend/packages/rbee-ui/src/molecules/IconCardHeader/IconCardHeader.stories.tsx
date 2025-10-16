import { Card, CardContent } from '@rbee/ui/atoms'
import type { Meta, StoryObj } from '@storybook/react'
import { Database, Globe, Lock, Shield, Zap } from 'lucide-react'
import { IconCardHeader } from './IconCardHeader'

const meta: Meta<typeof IconCardHeader> = {
  title: 'Molecules/IconCardHeader',
  component: IconCardHeader,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The IconCardHeader molecule provides a reusable card header pattern combining an icon, title, and optional subtitle. It composes IconPlate, CardTitle, and CardDescription in a standard layout.

## Composition
This molecule is composed of:
- **CardHeader**: Wrapper with mb-6 p-0
- **IconPlate**: Icon container with configurable size and tone
- **CardTitle**: Title text with configurable size
- **CardDescription**: Optional subtitle text

## When to Use
- Card headers that need an icon alongside title/subtitle
- Compliance cards, feature cards, status cards
- Any card that benefits from visual icon identification

## Used In
- **EnterpriseCompliance**: GDPR, SOC2, ISO 27001 compliance cards
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    icon: {
      control: false,
      description: 'Icon element',
      table: {
        type: { summary: 'ReactNode' },
        category: 'Content',
      },
    },
    title: {
      control: 'text',
      description: 'Card title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    subtitle: {
      control: 'text',
      description: 'Optional subtitle/description',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    titleId: {
      control: 'text',
      description: 'ID for the title (for aria-labelledby)',
      table: {
        type: { summary: 'string' },
        category: 'Accessibility',
      },
    },
    iconSize: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
      description: 'Icon size',
      table: {
        type: { summary: "'sm' | 'md' | 'lg'" },
        defaultValue: { summary: 'lg' },
        category: 'Appearance',
      },
    },
    iconTone: {
      control: 'select',
      options: ['primary', 'muted', 'success', 'warning'],
      description: 'Icon tone',
      table: {
        type: { summary: "'primary' | 'muted' | 'success' | 'warning'" },
        defaultValue: { summary: 'primary' },
        category: 'Appearance',
      },
    },
    titleClassName: {
      control: 'text',
      description: 'Title size class',
      table: {
        type: { summary: 'string' },
        defaultValue: { summary: 'text-2xl' },
        category: 'Appearance',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof IconCardHeader>

export const Default: Story = {
  args: {
    icon: <Globe className="size-6" />,
    title: 'GDPR',
    subtitle: 'EU Regulation',
    titleId: 'example-gdpr',
  },
}

export const WithoutSubtitle: Story = {
  args: {
    icon: <Zap className="size-6" />,
    title: 'Performance',
    titleId: 'example-performance',
  },
}

export const SmallIcon: Story = {
  args: {
    icon: <Shield className="size-6" />,
    title: 'Security',
    subtitle: 'Enterprise Grade',
    iconSize: 'sm',
    titleId: 'example-security',
  },
}

export const MutedTone: Story = {
  args: {
    icon: <Database className="size-6" />,
    title: 'Storage',
    subtitle: 'Encrypted at Rest',
    iconTone: 'muted',
    titleId: 'example-storage',
  },
}

export const CustomTitleSize: Story = {
  args: {
    icon: <Lock className="size-6" />,
    title: 'ISO 27001',
    subtitle: 'International Standard',
    titleClassName: 'text-xl',
    titleId: 'example-iso',
  },
}

export const InCard: Story = {
  render: () => (
    <div className="w-96">
      <Card className="rounded-2xl border-border bg-card/60 p-8">
        <IconCardHeader
          icon={<Globe className="size-6" />}
          title="GDPR"
          subtitle="EU Regulation"
          titleId="card-example-gdpr"
        />
        <CardContent className="p-0">
          <p className="text-sm text-foreground/85">
            Built from the ground up to meet GDPR requirements with data processing agreements, right to erasure, and
            privacy by design.
          </p>
        </CardContent>
      </Card>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'IconCardHeader used within a Card component with content.',
      },
    },
  },
}

export const MultipleCards: Story = {
  render: () => (
    <div className="grid max-w-4xl gap-6 md:grid-cols-3">
      <Card className="rounded-2xl border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Globe className="size-6" />}
          title="GDPR"
          subtitle="EU Regulation"
          titleId="multi-gdpr"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">EU data protection compliance</p>
        </CardContent>
      </Card>

      <Card className="rounded-2xl border-border bg-card/60 p-6">
        <IconCardHeader icon={<Shield className="size-6" />} title="SOC2" subtitle="US Standard" titleId="multi-soc2" />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Security and availability controls</p>
        </CardContent>
      </Card>

      <Card className="rounded-2xl border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Lock className="size-6" />}
          title="ISO 27001"
          subtitle="International"
          titleId="multi-iso"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Information security management</p>
        </CardContent>
      </Card>
    </div>
  ),
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        story: 'Multiple cards using IconCardHeader in a grid layout, as seen in EnterpriseCompliance.',
      },
    },
  },
}
