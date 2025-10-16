import type { Meta, StoryObj } from '@storybook/react'
import { Eye, Lock, Shield } from 'lucide-react'
import { SecurityCrate } from './SecurityCrate'

const meta: Meta<typeof SecurityCrate> = {
  title: 'Molecules/SecurityCrate',
  component: SecurityCrate,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The SecurityCrate molecule displays a security crate's capabilities with icon, title, subtitle, intro, bullet points, and optional documentation link.

## Composition
This molecule is composed of:
- **IconPlate**: Large icon representing the crate
- **Title**: Crate name (e.g., "auth-min: Zero-Trust Authentication")
- **Subtitle**: Optional tagline (e.g., "The Trickster Guardians")
- **Intro**: Description paragraph
- **Bullets**: List of features/capabilities with CheckItem
- **Docs Link**: Optional link to documentation

## When to Use
- Showcasing security features
- Explaining security crates
- Technical security documentation
- Enterprise security pages

## Used In
- **EnterpriseSecurity**: Displays security crates (auth-min, audit-core, secrets-vault)
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    icon: {
      control: false,
      description: 'Lucide icon component (e.g., Lock, Shield, Eye)',
      table: {
        type: { summary: 'LucideIcon' },
        category: 'Content',
      },
    },
    title: {
      control: 'text',
      description: 'Crate title',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    subtitle: {
      control: 'text',
      description: 'Optional subtitle',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    intro: {
      control: 'text',
      description: 'Introduction paragraph',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    bullets: {
      control: 'object',
      description: 'List of features/capabilities',
      table: {
        type: { summary: 'string[]' },
        category: 'Content',
      },
    },
    docsHref: {
      control: 'text',
      description: 'Optional documentation link',
      table: {
        type: { summary: 'string' },
        category: 'Behavior',
      },
    },
    tone: {
      control: 'select',
      options: ['primary', 'neutral'],
      description: 'Visual tone',
      table: {
        type: { summary: "'primary' | 'neutral'" },
        defaultValue: { summary: 'primary' },
        category: 'Appearance',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof SecurityCrate>

export const Default: Story = {
  args: {
    icon: <Lock className="size-6" />,
    title: 'auth-min: Zero-Trust Authentication',
    subtitle: 'The Trickster Guardians',
    intro: 'Minimal authentication with maximum security. Every request is verified, every token is short-lived.',
    bullets: [
      'JWT with 15-minute expiry',
      'Refresh token rotation',
      'Device fingerprinting',
      'Rate limiting per endpoint',
    ],
    docsHref: '/docs/security/auth-min',
  },
}

export const WithIcon: Story = {
  args: {
    icon: <Shield className="size-6" />,
    title: 'audit-core: Compliance Logging',
    subtitle: 'The Watchers',
    intro: 'Every action is logged, every change is tracked. Immutable audit trails for compliance.',
    bullets: ['Structured JSON logs', 'Tamper-proof storage', 'Real-time alerting', 'Compliance exports'],
    docsHref: '/docs/security/audit-core',
  },
}

export const WithDetails: Story = {
  args: {
    icon: <Eye className="size-6" />,
    title: 'secrets-vault: Encrypted Secrets',
    subtitle: 'The Keepers',
    intro: 'Secrets are encrypted at rest and in transit. Access is logged and audited.',
    bullets: [
      'AES-256 encryption',
      'Key rotation policies',
      'Access control lists',
      'Audit trail for all reads',
      'Integration with KMS',
      'Zero-knowledge architecture',
    ],
    docsHref: '/docs/security/secrets-vault',
    tone: 'neutral',
  },
}

export const InSecurityContext: Story = {
  render: () => (
    <div className="w-full max-w-6xl">
      <div className="mb-4 text-sm text-muted-foreground">Example: SecurityCrate in EnterpriseSecurity organism</div>
      <div className="grid gap-6 md:grid-cols-3">
        <SecurityCrate
          icon={<Lock className="size-6" />}
          title="auth-min"
          subtitle="Zero-Trust Authentication"
          intro="Minimal authentication with maximum security. Every request is verified."
          bullets={['JWT with 15-minute expiry', 'Refresh token rotation', 'Device fingerprinting', 'Rate limiting']}
          docsHref="/docs/security/auth-min"
        />
        <SecurityCrate
          icon={<Shield className="size-6" />}
          title="audit-core"
          subtitle="Compliance Logging"
          intro="Every action is logged, every change is tracked. Immutable audit trails."
          bullets={['Structured JSON logs', 'Tamper-proof storage', 'Real-time alerting', 'Compliance exports']}
          docsHref="/docs/security/audit-core"
        />
        <SecurityCrate
          icon={<Eye className="size-6" />}
          title="secrets-vault"
          subtitle="Encrypted Secrets"
          intro="Secrets are encrypted at rest and in transit. Access is logged and audited."
          bullets={['AES-256 encryption', 'Key rotation policies', 'Access control lists', 'Audit trail']}
          docsHref="/docs/security/secrets-vault"
        />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'SecurityCrate as used in the EnterpriseSecurity organism, showcasing three security crates.',
      },
    },
  },
}
