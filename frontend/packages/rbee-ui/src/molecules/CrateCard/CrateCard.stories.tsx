import type { Meta, StoryObj } from '@storybook/react'
import { CrateCard } from './CrateCard'

const meta: Meta<typeof CrateCard> = {
  title: 'Molecules/CrateCard',
  component: CrateCard,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
## Overview
CrateCard is a lightweight molecule for displaying crate/package information in compact grid layouts.

## When to Use
- Security crate lattices
- Package/library grids
- Technology stack displays
- Dependency lists

## Composition
- **Name**: Bold crate/package name
- **Description**: Brief one-line description
- **Hover effect**: Customizable border color on hover

## Design
Minimal card with just name + description. No icons, no features list—just the essentials for dense information grids.
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    name: {
      control: 'text',
      description: 'Crate/package name',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    description: {
      control: 'text',
      description: 'Brief description',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    hoverColor: {
      control: 'text',
      description: 'Tailwind hover border color class',
      table: {
        type: { summary: 'string' },
        defaultValue: { summary: 'hover:border-primary/50' },
        category: 'Appearance',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof CrateCard>

export const Default: Story = {
  args: {
    name: 'auth-min',
    description: 'Timing-safe tokens, zero-trust auth.',
  },
}

export const WithCustomHoverColor: Story = {
  args: {
    name: 'audit-logging',
    description: 'Append-only logs, 7-year retention.',
    hoverColor: 'hover:border-chart-3/50',
  },
}

export const LongDescription: Story = {
  args: {
    name: 'input-validation',
    description: 'Comprehensive injection prevention, schema validation, and resource exhaustion protection.',
  },
}

export const SecurityCratesGrid: Story = {
  render: () => (
    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 max-w-4xl">
      <CrateCard
        name="auth-min"
        description="Timing-safe tokens, zero-trust auth."
        hoverColor="hover:border-chart-2/50"
      />
      <CrateCard
        name="audit-logging"
        description="Append-only logs, 7-year retention."
        hoverColor="hover:border-chart-3/50"
      />
      <CrateCard
        name="input-validation"
        description="Injection prevention, schema validation."
        hoverColor="hover:border-primary/50"
      />
      <CrateCard
        name="secrets-management"
        description="Encrypted storage, rotation, KMS-friendly."
        hoverColor="hover:border-amber-500/50"
      />
      <CrateCard
        name="jwt-guardian"
        description="RS256 validation, revocation lists, short-lived tokens."
        hoverColor="hover:border-chart-2/50"
      />
      <CrateCard
        name="deadline-propagation"
        description="Timeouts, cleanup, cascading shutdown."
        hoverColor="hover:border-chart-3/50"
      />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'Example of CrateCard used in a 3-column grid showing all 6 security crates with different hover colors.',
      },
    },
  },
}

export const TechnologyStack: Story = {
  render: () => (
    <div className="grid sm:grid-cols-2 gap-3 max-w-2xl">
      <CrateCard name="tokio" description="Async runtime for Rust." hoverColor="hover:border-primary/50" />
      <CrateCard name="axum" description="Web framework built on tokio." hoverColor="hover:border-chart-2/50" />
      <CrateCard name="serde" description="Serialization framework." hoverColor="hover:border-chart-3/50" />
      <CrateCard name="tracing" description="Application-level tracing." hoverColor="hover:border-chart-4/50" />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'CrateCard can be used for any technology stack or dependency listing.',
      },
    },
  },
}
