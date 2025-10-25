import type { Meta, StoryObj } from '@storybook/react'
import { BrandMark } from './BrandMark'

const meta: Meta<typeof BrandMark> = {
  title: 'Atoms/BrandMark',
  component: BrandMark,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The BrandMark component renders the rbee bee logo as an SVG icon. It's the core visual identifier of the rbee brand.

## Composition
This is a standalone SVG atom component that represents the bee icon portion of the brand identity.

## When to Use
- As part of the BrandLogo molecule (logo + wordmark)
- In favicon or app icons
- In loading states or minimal branding contexts
- Where space is constrained and wordmark cannot fit

## Variants
- **Small (sm)**: Compact size for tight spaces
- **Medium (md)**: Default size for most contexts
- **Large (lg)**: Prominent branding in hero sections

## Used In Commercial Site
- **BrandLogo Component**: Combined with wordmark in Navigation and Footer
- **Loading States**: Standalone bee icon during page transitions
- **Favicon**: Browser tab icon
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
      description: 'Size variant of the brand mark',
      table: {
        type: { summary: "'sm' | 'md' | 'lg'" },
        defaultValue: { summary: 'md' },
        category: 'Appearance',
      },
    },
    priority: {
      control: 'boolean',
      description: 'Whether to prioritize loading (for above-the-fold images)',
      table: {
        type: { summary: 'boolean' },
        defaultValue: { summary: 'false' },
        category: 'Performance',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof BrandMark>

export const Default: Story = {
  args: {
    size: 'md',
  },
}

export const Small: Story = {
  args: {
    size: 'sm',
  },
}

export const Large: Story = {
  args: {
    size: 'lg',
  },
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex items-center gap-8">
      <div className="flex flex-col items-center gap-2">
        <BrandMark size="sm" />
        <span className="text-sm text-muted-foreground">Small</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <BrandMark size="md" />
        <span className="text-sm text-muted-foreground">Medium</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <BrandMark size="lg" />
        <span className="text-sm text-muted-foreground">Large</span>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'All available size variants of the BrandMark. Used in different contexts depending on visual hierarchy needs.',
      },
    },
  },
}

export const InBrandLogo: Story = {
  render: () => (
    <div className="flex flex-col gap-6 p-8">
      <div className="text-sm text-muted-foreground">Example: BrandMark is used inside BrandLogo molecule</div>
      <div className="flex items-center gap-3 rounded border bg-card p-4">
        <BrandMark size="md" />
        <span className="text-xl font-bold text-foreground">rbee</span>
      </div>
      <div className="text-xs text-muted-foreground">See BrandLogo molecule for the complete implementation</div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'BrandMark is the core component of the BrandLogo molecule, combined with the wordmark text.',
      },
    },
  },
}
