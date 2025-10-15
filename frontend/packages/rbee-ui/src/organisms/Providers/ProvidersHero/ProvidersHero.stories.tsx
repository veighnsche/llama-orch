import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersHero } from './ProvidersHero'

const meta = {
  title: 'Organisms/Providers/ProvidersHero',
  component: ProvidersHero,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The ProvidersHero is the primary hero section for the GPU Providers page, targeting GPU owners who want to monetize their idle hardware. It features an earnings-focused value proposition, income statistics, and an interactive earnings dashboard visualization.

## Two-Sided Marketplace Strategy

### Provider Side
- **Target:** GPU owners (gamers, miners, homelabbers, small datacenters) with idle hardware
- **Value prop:** Turn idle GPUs into passive income (€50-200/month per GPU)
- **Pain points:** Expensive hardware sitting idle 80% of the time, wasted investment, high electricity costs
- **Objections:** Security concerns (will workloads access my files?), complexity (is it hard to set up?), payment trust (will I actually get paid?)

### Consumer Side
- **Value prop:** Access to distributed GPU capacity at competitive rates
- **Pain points:** High cloud costs, limited GPU availability, vendor lock-in

### Marketplace Mechanics
- **Discovery:** Providers list GPUs with pricing, consumers browse marketplace
- **Pricing:** Provider-set hourly rates with dynamic market pricing suggestions
- **Trust:** Reputation system, secure payouts, sandboxed workloads
- **Economics:** Platform takes small percentage, providers keep majority of earnings

## Composition
This organism contains:
- **Badge**: "Turn Idle GPUs Into Income" kicker
- **Headline**: "Your GPUs Can Pay You Every Month" - direct earnings focus
- **Value Proposition**: Join marketplace, set price, earn automatically
- **StatsGrid**: Three key stats (€50-200/month, 24/7 passive income, 100% secure payouts)
- **CTA Buttons**: Primary "Start Earning" and secondary "Estimate My Payout"
- **Trust Row**: "No tech expertise needed • Set your own prices • Pause anytime"
- **Earnings Dashboard Visual**: Interactive card showing monthly earnings, GPU list, utilization stats

## When to Use
- As the first section on the GPU Providers page (/gpu-providers)
- To immediately communicate earning potential to GPU owners
- To differentiate from consumer-focused messaging on other pages

## Content Requirements
- **Headline**: Earnings-focused, direct benefit statement
- **Stats**: Realistic earning ranges (€50-200/month per GPU)
- **Trust Signals**: Address security, complexity, payment concerns
- **Visual Proof**: Dashboard showing real earnings breakdown
- **CTAs**: "Start Earning" (primary) and "Estimate My Payout" (calculator link)

## Marketing Strategy
- **Target Audience:** GPU owners with idle hardware (gamers who work 9-5, miners seeking better ROI, homelabbers offsetting costs)
- **Primary Message:** "Your expensive GPU can pay for itself every month"
- **Emotional Appeal:** Frustration (wasted investment) → Freedom (passive income)
- **CTAs:** 
  - Primary: "Start Earning" - direct action to onboard as provider
  - Secondary: "Estimate My Payout" - calculator to show personalized earnings
- **Copy Tone:** Empowering, transparent, trust-building (addresses FUD directly)

## Variants
- **Default**: Full earnings focus with dashboard visual
- **EarningsFocus**: Emphasize income potential, larger stats
- **EasySetupFocus**: Emphasize simplicity, "no tech expertise needed"

## Examples
\`\`\`tsx
import { ProvidersHero } from '@rbee/ui/organisms/Providers/ProvidersHero'

// Simple usage - no props needed
<ProvidersHero />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- Badge
- Button
- StatsGrid
- ProgressBar

## Accessibility
- **Keyboard Navigation**: All buttons are keyboard accessible
- **ARIA Labels**: Proper labels on CTAs ("Start earning with rbee")
- **Semantic HTML**: Uses <section> with proper heading hierarchy
- **Color Contrast**: Meets WCAG AA standards in both themes
- **Focus States**: Visible focus indicators on interactive elements
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersHero>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersHero as used on /gpu-providers page.
 * Shows earnings-focused messaging with dashboard visual.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing earning potential.
 * Highlights the €50-200/month range and passive income angle.
 */
export const EarningsFocus: Story = {}

/**
 * Variant emphasizing ease of setup.
 * Focuses on "no tech expertise needed" and simplicity.
 */
export const EasySetupFocus: Story = {}
