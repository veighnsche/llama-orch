// Dark Mode Polish Showcase for Button (Links & States)
import type { Meta, StoryObj } from '@storybook/react'
import { ExternalLink } from 'lucide-react'
import { Button } from './Button'

const meta: Meta<typeof Button> = {
  title: 'Atoms/Button/Dark Mode Polish',
  component: Button,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#0b1220' }],
    },
  },
  decorators: [
    (Story) => (
      <div className="dark">
        <Story />
      </div>
    ),
  ],
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Button>

/**
 * ## Dark Mode Polish: Links & Visited States
 *
 * Demonstrates:
 * - Link default: text-[color:var(--accent)] (#d97706)
 * - Link hover: text-white + decoration-amber-300
 * - Link visited: text-[#b45309] (brand 700)
 * - Link focus: Proper ring with CSS variables
 */

export const LinkStates: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded space-y-8 max-w-2xl">
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Link Variants</h3>
        <div className="space-y-3">
          <div>
            <Button variant="link" asChild>
              <a href="#default">Default Link (#d97706)</a>
            </Button>
            <p className="text-xs text-muted-foreground mt-1">Hover to see text-white + decoration-amber-300</p>
          </div>

          <div>
            <Button variant="link" asChild>
              <a href="#visited" className="visited:text-[#b45309]">
                Visited Link (click to mark visited)
              </a>
            </Button>
            <p className="text-xs text-muted-foreground mt-1">After clicking, shows brand 700 (#b45309)</p>
          </div>

          <div>
            <Button variant="link" asChild>
              <a href="#focus">Focus this link (Tab key)</a>
            </Button>
            <p className="text-xs text-muted-foreground mt-1">Focus ring uses CSS variables for consistency</p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Link with Icon</h3>
        <Button variant="link" asChild>
          <a href="#external" target="_blank" rel="noopener noreferrer">
            Explore details
            <ExternalLink className="ml-1" />
          </a>
        </Button>
        <p className="text-xs text-muted-foreground">
          Micro-tone: "Explore details" reads calmer than "Learn more" on dark
        </p>
      </div>

      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Link Matrix</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <p className="text-sm font-medium">Default State</p>
            <Button variant="link" asChild>
              <a href="#matrix-1">Link 1</a>
            </Button>
            <Button variant="link" asChild>
              <a href="#matrix-2">Link 2</a>
            </Button>
          </div>
          <div className="space-y-2">
            <p className="text-sm font-medium">Hover State</p>
            <Button variant="link" asChild className="text-white decoration-amber-300">
              <a href="#matrix-3">Hovered Link</a>
            </Button>
            <Button variant="link" asChild className="text-white decoration-amber-300">
              <a href="#matrix-4">Hovered Link</a>
            </Button>
          </div>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Link states showing default, hover, visited, and focus with proper contrast on dark.',
      },
    },
  },
}

export const DisabledStates: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded space-y-6 max-w-md">
      <h3 className="text-lg font-semibold mb-4">Disabled Button States</h3>

      <div className="space-y-3">
        <div>
          <Button disabled>Primary Disabled</Button>
          <p className="text-xs text-muted-foreground mt-1">bg-[#1a2435], text-[#6c7a90], border-[#223047]</p>
        </div>

        <div>
          <Button variant="outline" disabled>
            Outline Disabled
          </Button>
          <p className="text-xs text-muted-foreground mt-1">Same disabled colors for consistency</p>
        </div>

        <div>
          <Button variant="secondary" disabled>
            Secondary Disabled
          </Button>
          <p className="text-xs text-muted-foreground mt-1">Prevents "vanishing" controls on deep canvas</p>
        </div>

        <div>
          <Button variant="ghost" disabled>
            Ghost Disabled
          </Button>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Disabled button states with clear legibility on dark canvas.',
      },
    },
  },
}

export const ButtonProgression: Story = {
  render: () => (
    <div className="bg-[#0b1220] p-8 rounded space-y-6 max-w-md">
      <h3 className="text-lg font-semibold mb-4">Brand Progression</h3>

      <div className="space-y-4">
        <div className="space-y-2">
          <p className="text-sm font-medium">Primary Button States</p>
          <div className="flex gap-2">
            <Button size="sm">Default</Button>
            <Button size="sm" className="bg-[#d97706]">
              Hover
            </Button>
            <Button size="sm" className="bg-[#92400e]">
              Active
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">#b45309 → #d97706 → #92400e</p>
        </div>

        <div className="space-y-2">
          <p className="text-sm font-medium">Ghost Button (Neutral)</p>
          <div className="flex gap-2">
            <Button variant="ghost" size="sm">
              Default
            </Button>
            <Button variant="ghost" size="sm" className="bg-white/[0.04]">
              Hover
            </Button>
            <Button variant="ghost" size="sm" className="bg-white/[0.06]">
              Active
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">Neutral hover (not amber) for data density contexts</p>
        </div>
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'Button state progression showing brand colors and neutral ghost variant.',
      },
    },
  },
}
