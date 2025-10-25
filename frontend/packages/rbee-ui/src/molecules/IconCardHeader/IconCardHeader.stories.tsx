import { Card, CardContent } from '@rbee/ui/atoms'
import type { Meta, StoryObj } from '@storybook/react'
import { Database, Globe, Lock, Shield, TrendingUp, Zap } from 'lucide-react'
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
    <Card className="w-96 rounded border-border bg-card/60 p-8">
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
      <Card className="rounded border-border bg-card/60 p-6">
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

      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader icon={<Shield className="size-6" />} title="SOC2" subtitle="US Standard" titleId="multi-soc2" />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Security and availability controls</p>
        </CardContent>
      </Card>

      <Card className="rounded border-border bg-card/60 p-6">
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

export const RealWorldUsage: Story = {
  render: () => (
    <div className="space-y-8 max-w-6xl">
      {/* EnterpriseCompliance Pattern - Uses CardHeader wrapper */}
      <div>
        <h3 className="mb-4 text-lg font-semibold">EnterpriseCompliance Pattern (with CardHeader)</h3>
        <div className="grid gap-6 lg:grid-cols-3">
          <Card className="h-full rounded border-border bg-card/60 p-8">
            <IconCardHeader
              icon={<Globe className="size-6" />}
              title="GDPR"
              subtitle="EU Regulation"
              titleId="real-gdpr"
            />
            <CardContent className="p-0">
              <ul className="space-y-3">
                <li className="text-sm">Data processing agreements</li>
                <li className="text-sm">Right to erasure</li>
                <li className="text-sm">Privacy by design</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="h-full rounded border-border bg-card/60 p-8">
            <IconCardHeader
              icon={<Shield className="size-6" />}
              title="SOC 2"
              subtitle="US Standard"
              titleId="real-soc2"
            />
            <CardContent className="p-0">
              <ul className="space-y-3">
                <li className="text-sm">Security controls</li>
                <li className="text-sm">Availability monitoring</li>
                <li className="text-sm">Annual audits</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="h-full rounded border-border bg-card/60 p-8">
            <IconCardHeader
              icon={<Lock className="size-6" />}
              title="ISO 27001"
              subtitle="International"
              titleId="real-iso"
            />
            <CardContent className="p-0">
              <ul className="space-y-3">
                <li className="text-sm">Information security</li>
                <li className="text-sm">Risk management</li>
                <li className="text-sm">Continuous improvement</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* SecurityIsolationTemplate Pattern */}
      <div>
        <h3 className="mb-4 text-lg font-semibold">SecurityIsolationTemplate Pattern</h3>
        <div className="grid gap-6 md:grid-cols-2">
          <Card className="p-6">
            <IconCardHeader
              icon={<Lock className="w-6 h-6" />}
              iconTone="chart-3"
              iconSize="sm"
              title="Process Isolation"
              subtitle="Sandboxed Execution"
              titleClassName="text-lg"
              subtitleClassName="text-sm mt-1"
            />
            <CardContent className="p-0">
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Separate process per request</li>
                <li>• Memory isolation</li>
                <li>• Resource limits enforced</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="p-6">
            <IconCardHeader
              icon={<Shield className="w-6 h-6" />}
              iconTone="chart-2"
              iconSize="sm"
              title="Zero-Trust Network"
              subtitle="Every Request Verified"
              titleClassName="text-lg"
              subtitleClassName="text-sm mt-1"
            />
            <CardContent className="p-0">
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• No implicit trust</li>
                <li>• Token-based auth</li>
                <li>• Continuous verification</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* ProvidersEarnings Pattern */}
      <div>
        <h3 className="mb-4 text-lg font-semibold">ProvidersEarnings Pattern</h3>
        <Card className="bg-gradient-to-b from-card to-background shadow-lg p-6">
          <IconCardHeader
            icon={<TrendingUp />}
            title="Your Earnings"
            iconSize="md"
            iconTone="primary"
            titleClassName="text-xl"
          />
          <CardContent className="p-0 space-y-4">
            <div className="text-4xl font-bold text-primary">€2,400</div>
            <p className="text-sm text-muted-foreground">Based on 20h/day at 80% utilization</p>
          </CardContent>
        </Card>
      </div>
    </div>
  ),
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        story:
          'Real-world usage patterns from actual templates: EnterpriseCompliance, SecurityIsolationTemplate, and ProvidersEarnings.',
      },
    },
  },
}

export const AllVariants: Story = {
  render: () => (
    <div className="grid max-w-6xl gap-6 md:grid-cols-2 lg:grid-cols-3">
      {/* Short title, short subtitle */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader icon={<Globe className="size-6" />} title="GDPR" subtitle="EU Regulation" titleId="variant-1" />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Short title and subtitle example</p>
        </CardContent>
      </Card>

      {/* Long title, short subtitle */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Shield className="size-6" />}
          title="ISO 27001 Information Security Management"
          subtitle="International Standard"
          titleId="variant-2"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Long title with short subtitle</p>
        </CardContent>
      </Card>

      {/* Short title, long subtitle */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Lock className="size-6" />}
          title="SOC 2"
          subtitle="System and Organization Controls for Service Organizations: Trust Services Criteria"
          titleId="variant-3"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Short title with long subtitle</p>
        </CardContent>
      </Card>

      {/* Long title, long subtitle */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Database className="size-6" />}
          title="Payment Card Industry Data Security Standard Compliance"
          subtitle="Comprehensive security requirements for organizations that handle credit card information and transactions"
          titleId="variant-4"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Long title and long subtitle</p>
        </CardContent>
      </Card>

      {/* No subtitle */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader icon={<Zap className="size-6" />} title="Performance Monitoring" titleId="variant-5" />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Title only, no subtitle</p>
        </CardContent>
      </Card>

      {/* Small icon */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Shield className="size-6" />}
          title="Security Audit"
          subtitle="Annual Review"
          iconSize="sm"
          titleId="variant-6"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Small icon size</p>
        </CardContent>
      </Card>

      {/* Medium icon */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Database className="size-6" />}
          title="Data Storage"
          subtitle="Encrypted"
          iconSize="md"
          titleId="variant-7"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Medium icon size</p>
        </CardContent>
      </Card>

      {/* Muted tone */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Globe className="size-6" />}
          title="Regional Compliance"
          subtitle="Multi-jurisdiction"
          iconTone="muted"
          titleId="variant-8"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Muted icon tone</p>
        </CardContent>
      </Card>

      {/* Custom title size */}
      <Card className="rounded border-border bg-card/60 p-6">
        <IconCardHeader
          icon={<Lock className="size-6" />}
          title="Encryption"
          subtitle="AES-256"
          titleClassName="text-xl"
          titleId="variant-9"
        />
        <CardContent className="p-0">
          <p className="text-sm text-muted-foreground">Custom title size (text-xl)</p>
        </CardContent>
      </Card>
    </div>
  ),
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        story:
          'Comprehensive showcase of all IconCardHeader variants including different title/subtitle lengths, icon sizes, tones, and configurations.',
      },
    },
  },
}
