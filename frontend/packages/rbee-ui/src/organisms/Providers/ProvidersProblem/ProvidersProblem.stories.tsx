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
export const ProvidersPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story: `Default ProvidersProblem as used on /gpu-providers page. Shows all three provider pain points with financial impact tags.

**Problem 1: Wasted Investment**
- **Icon**: TrendingDown (red/destructive)
- **Tone**: Destructive (critical financial waste)
- **Copy**: "You paid €1,500+ for a high-end GPU. It's busy maybe 10% of the time—the other 90% earns €0."
- **Tag**: "Potential earnings €50-200/mo"
- **Target**: GPU owners with idle hardware (gamers, ML enthusiasts, homelab builders)
- **Why this pain point**: This addresses the sunk cost fallacy. You spent €1,500+ on a GPU, but it's idle 90% of the time. The copywriter chose "€1,500+" because that's the realistic cost of a high-end GPU (RTX 4090, RTX 4080). "Busy maybe 10% of the time" is based on real usage patterns—most GPUs sit idle. The tag "Potential earnings €50-200/mo" quantifies the opportunity cost. This pain point was chosen first because it's the most emotionally resonant—you spent a lot of money, and it's doing nothing.

**Problem 2: Electricity Costs**
- **Icon**: Zap (red/destructive)
- **Tone**: Destructive (direct financial loss)
- **Copy**: "Idle GPUs still pull power. That's roughly €10-30 each month spent on doing nothing."
- **Tag**: "Direct loss €10-30/mo"
- **Target**: GPU owners concerned about electricity costs
- **Why this pain point**: This addresses the ongoing cost of idle hardware. Idle GPUs still draw power (50-100W at idle). At €0.30/kWh (EU average), that's €10-30/month. The copywriter chose "spent on doing nothing" to emphasize the waste—you're paying for electricity to power hardware that's earning nothing. The tag "Direct loss €10-30/mo" makes the cost concrete. This pain point was chosen second because it's a recurring cost that adds up over time.

**Problem 3: Missed Opportunity**
- **Icon**: AlertCircle (red/destructive)
- **Tone**: Destructive (opportunity cost)
- **Copy**: "Developers rent GPU power every day. Your machine could join the marketplace and get paid automatically."
- **Tag**: "Unrealized €50-200/mo"
- **Target**: GPU owners who don't know about GPU rental marketplaces
- **Why this pain point**: This addresses the opportunity cost of not participating in the GPU rental market. Developers rent GPU power every day (RunPod, Vast.ai, Lambda Labs). Your idle GPU could be earning €50-200/month. The copywriter chose "get paid automatically" to emphasize the passive income aspect—you don't have to do anything, the marketplace handles it. The tag "Unrealized €50-200/mo" quantifies the opportunity. This pain point was chosen last because it presents the solution—join the marketplace and start earning.`,
			},
		},
	},
}

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
