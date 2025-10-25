import type { Meta, StoryObj } from '@storybook/react'
import { GlassCard } from './GlassCard'

const meta = {
  title: 'Atoms/GlassCard',
  component: GlassCard,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
GlassCard is an atom component that provides a frosted glass effect (glassmorphism) with backdrop blur and semi-transparent background. Perfect for floating overlays, badges, and modern UI elements.

## Features
- **Backdrop blur**: Creates frosted glass effect
- **Semi-transparent background**: Adapts to light/dark themes
- **Rounded corners**: Modern, polished appearance
- **Shadow**: Subtle depth and elevation
- **Full HTML support**: Extends HTMLDivElement for all standard attributes

## When to Use
- Floating KPI cards or badges
- Overlay elements that need to show content beneath
- Modern, premium UI elements
- Status indicators with glass effect

## Accessibility
- Supports all standard HTML attributes including ARIA
- Can be used with \`role\`, \`aria-live\`, \`aria-label\`, etc.

## Examples
\`\`\`tsx
import { GlassCard } from '@rbee/ui/atoms'

// Simple usage
<GlassCard>
  <p>Content with glass effect</p>
</GlassCard>

// With ARIA attributes
<GlassCard role="status" aria-live="polite" aria-label="Live updates">
  <div>Status: Active</div>
</GlassCard>

// Floating badge
<GlassCard className="absolute -top-4 -right-4">
  <div className="text-xs text-muted-foreground">Label</div>
  <div className="font-semibold text-primary">Value</div>
</GlassCard>
\`\`\`

## Used In
- FloatingKPICard molecule
- EnterpriseHero floating badges
- Any component requiring glass effect
        `,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof GlassCard>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    children: (
      <div className="p-4">
        <div className="text-sm text-muted-foreground">Label</div>
        <div className="text-lg font-semibold text-primary">Value</div>
      </div>
    ),
  },
  parameters: {
    docs: {
      description: {
        story:
          'Default GlassCard with simple content. Shows the frosted glass effect with backdrop blur and semi-transparent background.',
      },
    },
  },
}

export const FloatingBadge: Story = {
  args: {
    children: (
      <div className="px-4 py-2">
        <div className="text-xs text-muted-foreground">Data Residency</div>
        <div className="font-semibold text-primary">EU Only</div>
      </div>
    ),
  },
  parameters: {
    docs: {
      description: {
        story: 'GlassCard styled as a floating badge. Compact padding and typography for status indicators.',
      },
    },
  },
}

export const WithIcon: Story = {
  args: {
    children: (
      <div className="flex items-center gap-3 p-4">
        <div className="h-10 w-10 rounded-full bg-primary/20 flex items-center justify-center">
          <svg className="h-5 w-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <div>
          <div className="text-sm text-muted-foreground">Status</div>
          <div className="font-semibold text-foreground">Compliant</div>
        </div>
      </div>
    ),
  },
  parameters: {
    docs: {
      description: {
        story: 'GlassCard with icon and text content. Useful for status cards with visual indicators.',
      },
    },
  },
}

export const KPICard: Story = {
  args: {
    children: (
      <div className="p-4 space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">GPU Pool</span>
          <span className="font-semibold">5 nodes / 8 GPUs</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Cost</span>
          <span className="font-semibold text-emerald-500">$0.00 / hr</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Latency</span>
          <span className="font-semibold">~34 ms</span>
        </div>
      </div>
    ),
  },
  parameters: {
    docs: {
      description: {
        story:
          'GlassCard as a KPI card showing multiple metrics. Demonstrates use case for floating performance indicators.',
      },
    },
  },
}

export const WithBackground: Story = {
  args: {
    children: (
      <div className="p-6">
        <div className="text-sm text-muted-foreground mb-1">Audit Events</div>
        <div className="text-2xl font-bold text-primary">32 Types</div>
      </div>
    ),
  },
  decorators: [
    (Story) => (
      <div className="relative w-[400px] h-[300px] bg-gradient-to-br from-primary/20 to-secondary/20 rounded p-8">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENvbG9yIiBzdHJva2Utb3BhY2l0eT0iMC4xIiBzdHJva2Utd2lkdGg9IjEiLz48L3BhdHRlcm4+PC9kZWZzPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9InVybCgjZ3JpZCkiLz48L3N2Zz4=')] opacity-20" />
        <Story />
      </div>
    ),
  ],
  parameters: {
    docs: {
      description: {
        story:
          'GlassCard over a colorful background. Demonstrates how the frosted glass effect allows background content to show through while maintaining readability.',
      },
    },
  },
}

export const Compact: Story = {
  args: {
    children: (
      <div className="px-3 py-1.5">
        <span className="text-xs font-medium text-primary">NEW</span>
      </div>
    ),
  },
  parameters: {
    docs: {
      description: {
        story: 'Compact GlassCard for small badges or labels. Minimal padding for tight spaces.',
      },
    },
  },
}
