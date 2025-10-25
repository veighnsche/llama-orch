import type { Meta, StoryObj } from '@storybook/react'
import { NavLink } from './NavLink'

const meta: Meta<typeof NavLink> = {
  title: 'Molecules/NavLink',
  component: NavLink,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
NavLink is a navigation link molecule with active state detection and hover effects. It uses Next.js routing and automatically highlights the current page.

## Composition
This molecule is composed of:
- **Next.js Link**: Client-side navigation
- **Active state**: Automatic detection via usePathname
- **Underline indicator**: Animated underline for active state
- **Hover effects**: Smooth color transitions

## When to Use
- Main navigation (header links)
- Mobile navigation (drawer links)
- Footer navigation (footer links)
- Breadcrumbs (navigation trails)
- Tab navigation (section tabs)

## Variants
- **default**: Desktop navigation with underline
- **mobile**: Mobile navigation with larger text

## Used In Commercial Site
Used in:
- Navigation (header navigation)
- MobileNavigation (mobile drawer)
- Footer (footer links)
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    href: {
      control: 'text',
      description: 'Link destination',
      table: {
        type: { summary: 'string' },
        category: 'Content',
      },
    },
    children: {
      control: 'text',
      description: 'Link text',
      table: {
        type: { summary: 'ReactNode' },
        category: 'Content',
      },
    },
    variant: {
      control: 'select',
      options: ['default', 'mobile'],
      description: 'Visual variant',
      table: {
        type: { summary: "'default' | 'mobile'" },
        defaultValue: { summary: 'default' },
        category: 'Appearance',
      },
    },
    target: {
      control: 'text',
      description: 'Link target (e.g., "_blank")',
      table: {
        type: { summary: 'string' },
        category: 'Behavior',
      },
    },
    rel: {
      control: 'text',
      description: 'Link relationship (e.g., "noopener noreferrer")',
      table: {
        type: { summary: 'string' },
        category: 'Behavior',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof NavLink>

export const Default: Story = {
  args: {
    href: '/features',
    children: 'Features',
    variant: 'default',
  },
}

export const Active: Story = {
  render: () => (
    <div className="flex gap-6 p-4 bg-card rounded border">
      <NavLink href="/" variant="default">
        Home
      </NavLink>
      <NavLink href="/features" variant="default">
        Features
      </NavLink>
      <NavLink href="/pricing" variant="default">
        Pricing
      </NavLink>
      <NavLink href="/docs" variant="default">
        Docs
      </NavLink>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Navigation links showing hover and active states. The active link has an underline indicator.',
      },
    },
  },
}

export const WithIcon: Story = {
  render: () => (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold mb-3">Desktop Navigation</h3>
        <div className="flex gap-6 p-4 bg-card rounded border">
          <NavLink href="/" variant="default">
            Home
          </NavLink>
          <NavLink href="/features" variant="default">
            Features
          </NavLink>
          <NavLink href="/developers" variant="default">
            Developers
          </NavLink>
          <NavLink href="/enterprise" variant="default">
            Enterprise
          </NavLink>
          <NavLink href="/pricing" variant="default">
            Pricing
          </NavLink>
        </div>
      </div>
      <div>
        <h3 className="text-sm font-semibold mb-3">Mobile Navigation</h3>
        <div className="flex flex-col gap-4 p-4 bg-card rounded border">
          <NavLink href="/" variant="mobile">
            Home
          </NavLink>
          <NavLink href="/features" variant="mobile">
            Features
          </NavLink>
          <NavLink href="/developers" variant="mobile">
            Developers
          </NavLink>
          <NavLink href="/enterprise" variant="mobile">
            Enterprise
          </NavLink>
          <NavLink href="/pricing" variant="mobile">
            Pricing
          </NavLink>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Both desktop and mobile navigation variants.',
      },
    },
  },
}

export const InNavigationContext: Story = {
  render: () => (
    <div className="w-full">
      <div className="mb-4 text-sm text-muted-foreground">Example: NavLink in Navigation component</div>
      <div className="w-full bg-card rounded border">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="font-bold text-lg">rbee</div>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <NavLink href="/" variant="default">
              Home
            </NavLink>
            <NavLink href="/features" variant="default">
              Features
            </NavLink>
            <NavLink href="/developers" variant="default">
              Developers
            </NavLink>
            <NavLink href="/enterprise" variant="default">
              Enterprise
            </NavLink>
            <NavLink href="/pricing" variant="default">
              Pricing
            </NavLink>
          </nav>
          <div className="flex items-center gap-3">
            <button className="px-4 py-2 text-sm font-semibold text-foreground hover:text-primary transition-colors">
              Sign In
            </button>
            <button className="px-4 py-2 text-sm font-semibold bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors">
              Get Started
            </button>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'NavLink as used in the Navigation component header.',
      },
    },
  },
}
