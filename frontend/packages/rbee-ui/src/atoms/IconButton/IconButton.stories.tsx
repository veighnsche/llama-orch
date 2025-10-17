// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { ChevronLeft, ChevronRight, Menu, Minus, Plus, Search, Settings, X } from 'lucide-react'
import { IconButton } from './IconButton'

const meta: Meta<typeof IconButton> = {
  title: 'Atoms/IconButton',
  component: IconButton,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
An icon-only button component for actions that don't need text labels.

## Features
- Consistent 36px (size-9) square footprint
- Hover and focus states with proper accessibility
- Supports asChild pattern for custom elements
- Disabled state support

## Used In
- Navigation components
- Toolbars and action bars
- Modal headers
- Mobile menus
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    asChild: {
      control: 'boolean',
      description: 'Render as a child element instead of button',
    },
    disabled: {
      control: 'boolean',
      description: 'Disable the button',
    },
    children: {
      control: false,
      description: 'Icon element to display',
    },
  },
}

export default meta
type Story = StoryObj<typeof IconButton>

export const Default: Story = {
  args: {
    children: <Menu />,
  },
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex items-center gap-4">
      <div className="flex flex-col items-center gap-2">
        <IconButton className="size-6">
          <Settings className="size-3" />
        </IconButton>
        <span className="text-xs text-muted-foreground">Small (24px)</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <IconButton className="size-8">
          <Settings className="size-4" />
        </IconButton>
        <span className="text-xs text-muted-foreground">Medium (32px)</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <IconButton>
          <Settings className="size-5" />
        </IconButton>
        <span className="text-xs text-muted-foreground">Default (36px)</span>
      </div>
      <div className="flex flex-col items-center gap-2">
        <IconButton className="size-12">
          <Settings className="size-6" />
        </IconButton>
        <span className="text-xs text-muted-foreground">Large (48px)</span>
      </div>
    </div>
  ),
}

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-col gap-6">
      <div>
        <h4 className="text-sm font-semibold mb-3">Common Actions</h4>
        <div className="flex gap-2">
          <IconButton>
            <Menu />
          </IconButton>
          <IconButton>
            <Search />
          </IconButton>
          <IconButton>
            <Settings />
          </IconButton>
          <IconButton>
            <X />
          </IconButton>
        </div>
      </div>
      <div>
        <h4 className="text-sm font-semibold mb-3">Navigation</h4>
        <div className="flex gap-2">
          <IconButton>
            <ChevronLeft />
          </IconButton>
          <IconButton>
            <ChevronRight />
          </IconButton>
        </div>
      </div>
      <div>
        <h4 className="text-sm font-semibold mb-3">Increment/Decrement</h4>
        <div className="flex gap-2">
          <IconButton>
            <Plus />
          </IconButton>
          <IconButton>
            <Minus />
          </IconButton>
        </div>
      </div>
      <div>
        <h4 className="text-sm font-semibold mb-3">States</h4>
        <div className="flex gap-2">
          <IconButton>
            <Settings />
          </IconButton>
          <IconButton disabled>
            <Settings />
          </IconButton>
        </div>
        <div className="flex gap-2 mt-2 text-xs text-muted-foreground">
          <span>Default</span>
          <span>Disabled</span>
        </div>
      </div>
    </div>
  ),
}

export const InNavigation: Story = {
  render: () => (
    <div className="w-full max-w-4xl">
      <nav className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center gap-2">
          <IconButton>
            <Menu />
          </IconButton>
          <span className="text-lg font-semibold">rbee</span>
        </div>
        <div className="flex items-center gap-2">
          <IconButton>
            <Search />
          </IconButton>
          <IconButton>
            <Settings />
          </IconButton>
        </div>
      </nav>
      <div className="p-4">
        <p className="text-sm text-muted-foreground">
          IconButton is commonly used in navigation bars for menu toggles, search, and settings access.
        </p>
      </div>
    </div>
  ),
}
