import type { Meta, StoryObj } from '@storybook/react'
import { ThemeToggle } from './ThemeToggle'

const meta: Meta<typeof ThemeToggle> = {
  title: 'Molecules/ThemeToggle',
  component: ThemeToggle,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
ThemeToggle is a button molecule that toggles between light and dark themes. It uses next-themes for theme management and displays appropriate icons for each state.

## Composition
This molecule is composed of:
- **IconButton**: Base button component
- **Sun icon**: Displayed in dark mode
- **Moon icon**: Displayed in light mode
- **Theme hook**: next-themes useTheme hook
- **Hydration safety**: Prevents hydration mismatches

## When to Use
- Navigation (header theme toggle)
- Settings panels (theme preferences)
- User controls (accessibility options)
- Anywhere users need to switch themes

## Variants
- **Light mode**: Shows moon icon
- **Dark mode**: Shows sun icon
- **Loading state**: Shows sun icon during hydration

## Used In Commercial Site
Used in:
- Navigation (header right side)
- MobileNavigation (mobile drawer)
- Footer (optional theme control)
				`,
      },
    },
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ThemeToggle>

export const Default: Story = {}

export const LightMode: Story = {
  render: () => (
    <div className="p-6 bg-background rounded border">
      <div className="flex items-center gap-4">
        <span className="text-sm text-muted-foreground">Light mode active:</span>
        <ThemeToggle />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'ThemeToggle in light mode, showing moon icon.',
      },
    },
  },
}

export const DarkMode: Story = {
  render: () => (
    <div className="p-6 bg-background rounded border">
      <div className="flex items-center gap-4">
        <span className="text-sm text-muted-foreground">Dark mode active:</span>
        <ThemeToggle />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'ThemeToggle in dark mode, showing sun icon.',
      },
    },
    backgrounds: {
      default: 'dark',
    },
  },
}

export const InNavigationContext: Story = {
  render: () => (
    <div className="w-full">
      <div className="mb-4 text-sm text-muted-foreground">Example: ThemeToggle in Navigation component</div>
      <div className="w-full bg-card rounded border">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-3">
            <div className="font-bold text-lg">rbee</div>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Features
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Pricing
            </a>
            <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Docs
            </a>
          </nav>
          <div className="flex items-center gap-3">
            <ThemeToggle />
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
        story: 'ThemeToggle as used in the Navigation component, positioned in the header controls.',
      },
    },
  },
}
