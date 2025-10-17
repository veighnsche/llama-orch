// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { Badge } from './Badge'

const meta: Meta<typeof Badge> = {
  title: 'Atoms/Badge',
  component: Badge,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'accent', 'secondary', 'destructive', 'outline'],
      description: 'Visual style variant of the badge',
    },
    asChild: {
      control: 'boolean',
      description: 'Render as child component using Radix Slot',
    },
  },
}

export default meta
type Story = StoryObj<typeof Badge>

/**
 * ## Overview
 * Badge is a compact UI element used to display status, categories, or counts.
 * Built with class-variance-authority for consistent styling across variants.
 *
 * ## When to Use
 * - Display status indicators (Active, Pending, Error)
 * - Show categories or tags
 * - Highlight counts or notifications
 * - Label items in lists or cards
 *
 * ## Used In
 * - PricingTier (plan features)
 * - FeatureCard (status indicators)
 * - ModelCard (model tags)
 * - Navigation (notification counts)
 * - And 25+ other organisms
 */

export const Default: Story = {
  args: {
    children: 'Badge',
  },
}

export const AllVariants: Story = {
  render: () => (
    <div className="flex flex-wrap gap-3">
      <Badge variant="default">Default</Badge>
      <Badge variant="accent">Accent</Badge>
      <Badge variant="secondary">Secondary</Badge>
      <Badge variant="destructive">Destructive</Badge>
      <Badge variant="outline">Outline</Badge>
    </div>
  ),
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex flex-wrap items-center gap-3">
      <Badge className="text-[10px] px-1.5 py-0">Tiny</Badge>
      <Badge>Default</Badge>
      <Badge className="text-sm px-3 py-1">Large</Badge>
    </div>
  ),
}

export const InOrganisms: Story = {
  render: () => (
    <div className="flex flex-col gap-6 max-w-md">
      {/* Pricing context */}
      <div className="border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold">Professional Plan</h3>
          <Badge variant="accent">Popular</Badge>
        </div>
        <p className="text-sm text-muted-foreground">Perfect for growing teams</p>
      </div>

      {/* Feature list context */}
      <div className="border rounded-lg p-4">
        <h3 className="font-semibold mb-3">Features</h3>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-sm">GPU Acceleration</span>
            <Badge variant="secondary">Available</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">24/7 Support</span>
            <Badge variant="outline">Coming Soon</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm">Legacy API</span>
            <Badge variant="destructive">Deprecated</Badge>
          </div>
        </div>
      </div>

      {/* Tag list context */}
      <div className="border rounded-lg p-4">
        <h3 className="font-semibold mb-3">Model Tags</h3>
        <div className="flex flex-wrap gap-2">
          <Badge variant="accent">Featured</Badge>
          <Badge variant="outline">LLaMA 3</Badge>
          <Badge variant="outline">8B</Badge>
          <Badge variant="outline">Instruct</Badge>
          <Badge variant="secondary">GGUF</Badge>
        </div>
      </div>
    </div>
  ),
}
