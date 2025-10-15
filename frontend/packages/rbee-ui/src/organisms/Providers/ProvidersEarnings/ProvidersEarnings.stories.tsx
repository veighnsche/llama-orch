import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersEarnings } from './ProvidersEarnings'

const meta = {
	title: 'Organisms/Providers/ProvidersEarnings',
	component: ProvidersEarnings,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProvidersEarnings section is an interactive earnings calculator that allows potential providers to estimate their monthly income based on GPU model, hours available per day, and expected utilization. It features a 2-column layout with configuration inputs on the left and earnings display on the right.

## Two-Sided Marketplace Strategy

### Earnings Transparency
- **Base Rates:** €0.18-0.45/hr depending on GPU model (RTX 3070 to RTX 4090)
- **Commission:** 15% platform fee (85% take-home for providers)
- **Realistic Estimates:** Conservative calculations with disclaimer
- **Interactive:** Users can adjust parameters to see personalized earnings

### Trust Building Through Transparency
- **Visible Breakdown:** Hourly rate × hours × utilization = monthly earnings
- **Commission Disclosure:** 15% fee shown upfront, not hidden
- **Conservative Estimates:** "Actuals vary with demand, pricing, and availability"
- **Payout Details:** €25 minimum, weekly payouts

### Competitive Positioning
- **Base Rates:** Competitive with NiceHash, Vast.ai, RunPod
- **Take-Home:** 85% is industry-standard (vs. 70-80% on some platforms)
- **Transparency:** Full breakdown vs. opaque pricing on competitors

## Composition
This organism contains:
- **Header**: "Calculate Your Potential Earnings"
- **Left Column (Configuration)**:
  - GPU Selection (6 models: RTX 4090 to RTX 3070)
  - Quick Presets (Casual, Daily, Always On)
  - Hours Per Day Slider (1-24h)
  - Utilization Slider (10-100%)
- **Right Column (Earnings Display)**:
  - Monthly Earnings (large KPI)
  - Take-Home After 15% Commission
  - Daily and Yearly Earnings
  - Breakdown (hourly rate, hours/month, utilization, commission)
  - "Start Earning Now" CTA
- **Disclaimer**: Conservative estimates notice

## When to Use
- On the GPU Providers page as earnings calculator
- Linked from hero and other sections (#earnings-calculator anchor)
- To provide personalized earnings estimates

## Content Requirements
- **GPU Models**: Realistic base rates (€0.18-0.45/hr)
- **Commission**: 15% disclosed upfront
- **Disclaimer**: "Earnings are estimates... may vary"
- **Interactive**: Real-time calculation as user adjusts parameters

## Marketing Strategy
- **Target Audience:** Potential providers evaluating opportunity
- **Primary Message:** "See exactly what you could earn"
- **Emotional Appeal:** Excitement (potential income) + Confidence (transparent calculation)
- **CTAs:** "Start Earning Now" - convert after seeing personalized estimate
- **Copy Tone:** Transparent, realistic, empowering

## Variants
- **Default**: Calculator with RTX 4090 selected, 20h/day, 80% utilization
- **HighEndGPU**: RTX 4090 example with optimistic parameters
- **MidRangeGPU**: RTX 3070 example with realistic parameters

## Examples
\`\`\`tsx
import { ProvidersEarnings } from '@rbee/ui/organisms/Providers/ProvidersEarnings'

// Simple usage - no props needed (interactive component)
<ProvidersEarnings />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)
- Linked from multiple sections via #earnings-calculator anchor

## Related Components
- Slider (for hours and utilization inputs)
- Button (for CTA)
- ProvidersHero (links to calculator)
- ProvidersSolution (links to calculator)

## Accessibility
- **Keyboard Navigation**: All inputs are keyboard accessible
- **ARIA Labels**: Sliders have aria-label attributes
- **Live Regions**: Monthly earnings uses aria-live="polite"
- **Focus Management**: Proper focus indicators on all interactive elements
- **Screen Readers**: All values and labels are properly announced
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof ProvidersEarnings>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersEarnings calculator.
 * RTX 4090 selected, 20h/day, 80% utilization.
 */
export const ProvidersPageDefault: Story = {}

/**
 * High-end GPU example (RTX 4090).
 * Shows maximum earning potential with optimistic parameters.
 */
export const HighEndGPU: Story = {}

/**
 * Mid-range GPU example (RTX 3070).
 * Shows realistic earnings for more common hardware.
 */
export const MidRangeGPU: Story = {}
