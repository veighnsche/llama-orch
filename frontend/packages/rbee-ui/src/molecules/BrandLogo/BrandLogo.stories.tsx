import type { Meta, StoryObj } from '@storybook/react'
import { BrandLogo } from './BrandLogo'

const meta: Meta<typeof BrandLogo> = {
  title: 'Molecules/BrandLogo',
  component: BrandLogo,
  parameters: {
    layout: 'centered',
    backgrounds: {
      default: 'dark',
    },
    docs: {
      description: {
        component: `
## Overview
The BrandLogo component combines the BrandMark (bee icon) with the "rbee" wordmark to create the complete brand identity. 

**⚠️ Framework-Agnostic:** This component is now framework-agnostic and does NOT include navigation logic. Wrap it with your framework's Link component for navigation.

## Composition
This molecule is composed of:
- **BrandMark**: The bee icon SVG (atom)
- **Wordmark**: The "rbee" text styled with brand typography

## Usage Examples

### Next.js
\`\`\`tsx
import Link from 'next/link'
import { BrandLogo } from '@rbee/ui/molecules'

<Link href="/">
  <BrandLogo size="md" />
</Link>
\`\`\`

### React Router (Tauri, Vite)
\`\`\`tsx
import { Link } from 'react-router-dom'
import { BrandLogo } from '@rbee/ui/molecules'

<Link to="/">
  <BrandLogo size="md" />
</Link>
\`\`\`

### Static (No Link)
\`\`\`tsx
<BrandLogo size="md" />
\`\`\`

## When to Use
- In the Navigation component (top-left corner of all pages)
- In the Footer component (brand identity section)
- In loading states or splash screens
- Anywhere you need the complete brand identity

## Variants
- **Small (sm)**: Compact logo for mobile navigation or footer
- **Medium (md)**: Default size for desktop navigation
- **Large (lg)**: Prominent branding in hero sections or landing pages

## Used In
- **Commercial Site (Next.js)**: Navigation and Footer
- **Queen UI (Vite)**: AppSidebar
- **Keeper UI (Tauri)**: KeeperSidebar
        `,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
      description: 'Size variant of the logo',
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
    as: {
      control: 'select',
      options: ['div', 'span'],
      description: 'Wrapper element type. Use "div" or "span" for static logo, or wrap with your own Link component.',
      table: {
        type: { summary: "'div' | 'span'" },
        defaultValue: { summary: 'div' },
        category: 'Structure',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof BrandLogo>

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

export const AsSpan: Story = {
  args: {
    size: 'md',
    as: 'span',
  },
  parameters: {
    docs: {
      description: {
        story: 'Using span wrapper for inline display contexts.',
      },
    },
  },
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex flex-col gap-8 p-8">
      <div className="flex flex-col gap-4">
        <h3 className="text-lg font-semibold text-foreground">Small</h3>
        <BrandLogo size="sm" />
      </div>
      <div className="flex flex-col gap-4">
        <h3 className="text-lg font-semibold text-foreground">Medium (Default)</h3>
        <BrandLogo size="md" />
      </div>
      <div className="flex flex-col gap-4">
        <h3 className="text-lg font-semibold text-foreground">Large</h3>
        <BrandLogo size="lg" />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'All available size variants. Small is used in mobile navigation and footer, medium in desktop navigation, and large in hero sections.',
      },
    },
  },
}

export const NavigationContext: Story = {
  render: () => (
    <div className="w-full">
      <div className="mb-4 text-sm text-muted-foreground">
        Example: BrandLogo in Navigation component (wrap with Link in real usage)
      </div>
      <div className="flex items-center justify-between rounded border bg-card p-4">
        <a href="/" className="inline-block">
          <BrandLogo size="md" />
        </a>
        <div className="flex gap-6">
          <a href="#" className="text-sm text-muted-foreground hover:text-foreground">
            Features
          </a>
          <a href="#" className="text-sm text-muted-foreground hover:text-foreground">
            Pricing
          </a>
          <a href="#" className="text-sm text-muted-foreground hover:text-foreground">
            Docs
          </a>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'BrandLogo as it appears in the Navigation component. Wrapped in a link element, positioned top-left, medium size on desktop. In real usage, wrap with your framework\'s Link component.',
      },
    },
  },
}

export const FooterContext: Story = {
  render: () => (
    <div className="w-full">
      <div className="mb-4 text-sm text-muted-foreground">
        Example: BrandLogo in Footer component (wrap with Link in real usage)
      </div>
      <div className="rounded border bg-card p-8">
        <div className="flex flex-col gap-4">
          <a href="/" className="inline-block">
            <BrandLogo size="md" />
          </a>
          <p className="max-w-xs text-sm text-muted-foreground">
            Private LLM hosting in the Netherlands. GDPR-compliant, self-hosted AI infrastructure.
          </p>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story:
          'BrandLogo as it appears in the Footer component. Wrapped in a link element with brand description below. In real usage, wrap with your framework\'s Link component.',
      },
    },
  },
}
