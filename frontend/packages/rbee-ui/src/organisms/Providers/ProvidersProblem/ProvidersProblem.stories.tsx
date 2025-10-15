import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersProblem } from './ProvidersProblem'

const meta = {
	title: 'Organisms/Providers/ProvidersProblem',
	component: ProvidersProblem,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProvidersProblem section articulates the pain points of GPU owners with idle hardware. It uses the shared ProblemSection component with provider-specific messaging focused on wasted investment, electricity costs, and missed earning opportunities.

## Two-Sided Marketplace Strategy

### Provider Pain Points
- **Wasted Investment:** €1,500+ GPU sitting idle 90% of the time, earning €0
- **Electricity Costs:** €10-30/month spent on idle power draw
- **Missed Opportunity:** Unrealized €50-200/month in potential earnings

### Emotional Journey
- **Before:** Frustration (expensive hardware doing nothing, paying electricity for nothing)
- **After:** Empowerment (hardware pays for itself, passive income stream)

### Messaging Differences from Consumer Problem
- Consumer problem: "Cloud costs too high, vendor lock-in"
- Provider problem: "Hardware costs too high, no ROI on idle time"

## Composition
This organism contains:
- **Kicker**: "The Cost of Idle GPUs"
- **Title**: "Stop Letting Your Hardware Bleed Money"
- **Subtitle**: Context about 90% idle time
- **Problem Items**: Three cards with icons, titles, bodies, and loss tags
  1. Wasted Investment (€50-200/mo potential)
  2. Electricity Costs (€10-30/mo direct loss)
  3. Missed Opportunity (€50-200/mo unrealized)
- **CTAs**: "Start Earning" (primary) and "Estimate My Payout" (secondary)
- **CTA Copy**: Urgency message about idle hours = money left on table

## When to Use
- On the GPU Providers page after the hero section
- To articulate provider-specific pain points
- To create urgency around idle hardware costs

## Content Requirements
- **Problem Items**: Must be specific to providers (not consumers)
- **Financial Impact**: Quantify losses in €/month
- **Emotional Tone**: Frustration and urgency (money bleeding away)
- **CTAs**: Link to onboarding and earnings calculator

## Marketing Strategy
- **Target Audience:** GPU owners with idle hardware
- **Primary Message:** "Your idle GPU is costing you money every month"
- **Emotional Appeal:** Frustration (wasted investment) + Urgency (ongoing loss)
- **CTAs:** 
  - Primary: "Start Earning" - stop the bleeding
  - Secondary: "Estimate My Payout" - quantify the opportunity
- **Copy Tone:** Direct, urgent, financial focus (€ amounts visible)

## Variants
- **Default**: All three problem items with financial tags
- **ROIFocus**: Emphasize wasted investment angle
- **UpgradeFocus**: Frame as "fund your next GPU upgrade"

## Examples
\`\`\`tsx
import { ProvidersProblem } from '@rbee/ui/organisms/Providers/ProvidersProblem'

// Simple usage - no props needed
<ProvidersProblem />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- ProblemSection (shared base component)
- ProvidersHero
- ProvidersSolution

## Accessibility
- **Semantic HTML**: Uses proper heading hierarchy
- **Icon Labels**: Icons are decorative (aria-hidden)
- **Color Contrast**: Destructive tone meets WCAG AA standards
- **Keyboard Navigation**: All CTAs are keyboard accessible
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof ProvidersProblem>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersProblem as used on /gpu-providers page.
 * Shows all three provider pain points with financial impact tags.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing wasted investment and ROI.
 * Focuses on the €1,500+ hardware earning €0.
 */
export const ROIFocus: Story = {}

/**
 * Variant framing idle earnings as upgrade funding.
 * "Your current GPU could pay for your next GPU upgrade."
 */
export const UpgradeFocus: Story = {}
