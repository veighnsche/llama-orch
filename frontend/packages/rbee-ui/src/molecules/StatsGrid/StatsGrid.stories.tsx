import type { Meta, StoryObj } from '@storybook/react'
import { Clock, Shield, Users, Zap } from 'lucide-react'
import { StatsGrid } from './StatsGrid'

const meta: Meta<typeof StatsGrid> = {
  title: 'Molecules/StatsGrid',
  component: StatsGrid,
  parameters: {
    layout: 'padded',
    docs: {
      description: {
        component: `
## Overview
StatsGrid is a unified molecule for displaying statistics across the commercial site. It consolidates multiple stat display patterns (pills, tiles, cards, inline) into a single reusable component.

## Composition
This molecule is composed of:
- **IconPlate**: Optional icon container for each stat
- **Grid layout**: Responsive column system (2, 3, or 4 columns)
- **Stat items**: Value + label pairs with optional icons and help text

## When to Use
- Hero sections (showing key metrics)
- CTA sections (highlighting benefits)
- Testimonials (social proof numbers)
- Feature sections (performance metrics)

## Variants
- **pills**: Compact horizontal layout with icons
- **tiles**: Card-style tiles with prominent values
- **cards**: Centered layout with large values (default)
- **inline**: Minimal inline stats

## Used In Commercial Site
Used in 5+ organisms including:
- HeroSection (key metrics)
- CTASection (conversion stats)
- TestimonialsSection (social proof)
- FeaturesSection (performance numbers)
- ProvidersSection (earning potential)
				`,
      },
    },
  },
  tags: ['autodocs'],
  argTypes: {
    stats: {
      control: 'object',
      description: 'Array of stat items to display',
      table: {
        type: { summary: 'StatItem[]' },
        category: 'Content',
      },
    },
    variant: {
      control: 'select',
      options: ['pills', 'tiles', 'cards', 'inline'],
      description: 'Visual variant',
      table: {
        type: { summary: "'pills' | 'tiles' | 'cards' | 'inline'" },
        defaultValue: { summary: 'cards' },
        category: 'Appearance',
      },
    },
    columns: {
      control: 'select',
      options: [2, 3, 4],
      description: 'Number of columns (responsive)',
      table: {
        type: { summary: '2 | 3 | 4' },
        defaultValue: { summary: '3' },
        category: 'Layout',
      },
    },
  },
}

export default meta
type Story = StoryObj<typeof StatsGrid>

const sampleStats = [
  { value: '100%', label: 'GDPR Compliant', icon: <Shield className="size-6" /> },
  { value: '<50ms', label: 'Latency', icon: <Zap className="size-6" /> },
  { value: '24/7', label: 'Support', icon: <Clock className="size-6" /> },
]

const fourStats = [
  { value: '100%', label: 'GDPR Compliant', icon: <Shield className="size-6" /> },
  { value: '<50ms', label: 'Latency', icon: <Zap className="size-6" /> },
  { value: '24/7', label: 'Support', icon: <Clock className="size-6" /> },
  { value: '500+', label: 'Users', icon: <Users className="size-6" /> },
]

export const Default: Story = {
  args: {
    stats: sampleStats,
    variant: 'cards',
    columns: 3,
  },
}

export const TwoColumn: Story = {
  args: {
    stats: [
      { value: '€50-200/mo', label: 'Earn with your GPU' },
      { value: '100% GDPR', label: 'Dutch data sovereignty' },
    ],
    variant: 'cards',
    columns: 2,
  },
}

export const WithTones: Story = {
  render: () => (
    <div className="space-y-8">
      <div>
        <h3 className="text-lg font-semibold mb-4">Pills Variant</h3>
        <StatsGrid stats={sampleStats} variant="pills" columns={3} />
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-4">Tiles Variant</h3>
        <StatsGrid stats={sampleStats} variant="tiles" columns={3} />
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-4">Cards Variant (Default)</h3>
        <StatsGrid stats={sampleStats} variant="cards" columns={3} />
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-4">Inline Variant</h3>
        <StatsGrid stats={sampleStats} variant="inline" columns={3} />
      </div>
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'All four visual variants showing different stat display styles.',
      },
    },
  },
}

export const InHeroContext: Story = {
  render: () => (
    <div className="w-full bg-muted p-8 rounded-lg">
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold mb-2">Private LLM Hosting</h2>
        <p className="text-muted-foreground">GDPR-compliant AI infrastructure in the Netherlands</p>
      </div>
      <StatsGrid stats={fourStats} variant="pills" columns={4} />
    </div>
  ),
  parameters: {
    docs: {
      description: {
        story: 'StatsGrid as used in HeroSection, showing key metrics with pills variant.',
      },
    },
  },
}
