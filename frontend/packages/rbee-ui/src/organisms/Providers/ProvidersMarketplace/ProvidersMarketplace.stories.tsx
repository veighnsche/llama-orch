import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersMarketplace } from './ProvidersMarketplace'

const meta = {
	title: 'Organisms/Providers/ProvidersMarketplace',
	component: ProvidersMarketplace,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProvidersMarketplace section explains how the rbee marketplace works from the provider perspective. It features four key marketplace benefits (Dynamic Pricing, Growing Demand, Global Reach, Fair Commission) and detailed marketplace features with commission structure breakdown.

## Two-Sided Marketplace Strategy

### Marketplace Mechanics
- **Discovery:** Automatic matching based on GPU specs and pricing
- **Pricing:** Provider-set rates with optional auto-pricing suggestions
- **Trust:** Rating system, guaranteed payments, dispute resolution
- **Economics:** 15% platform fee, 85% provider take-home

### Cold-Start Problem Solutions
- **Growing Demand:** "Thousands of AI jobs posted monthly"
- **Global Reach:** "Your GPUs are discoverable worldwide"
- **Automatic Matching:** Jobs match your GPUs automatically
- **Guaranteed Payments:** Customers pre-pay, every completed job is paid

### Marketplace Liquidity
- **Provider Side:** Easy onboarding, flexible pricing, control over availability
- **Consumer Side:** Large GPU pool, competitive pricing, quality ratings
- **Platform:** Automatic matching, payment escrow, dispute resolution

## Composition
This organism contains:
- **Header**: "How the rbee Marketplace Works"
- **Feature Tiles** (4-column grid):
  1. Dynamic Pricing (TrendingUp icon)
  2. Growing Demand (Users icon)
  3. Global Reach (Globe icon)
  4. Fair Commission (Shield icon)
- **Features & Commission Split** (2-column layout):
  - **Left:** Marketplace Features (Automatic Matching, Rating System, Guaranteed Payments, Dispute Resolution)
  - **Right:** Commission Structure (15% standard, 85% you keep, example breakdown)

## When to Use
- On the GPU Providers page to explain marketplace mechanics
- To address provider concerns about payment and trust
- To differentiate from competitor platforms

## Content Requirements
- **Marketplace Features:** Must address provider concerns (matching, payments, disputes)
- **Commission Structure:** Transparent breakdown with example
- **Trust Signals:** Guaranteed payments, rating system, dispute resolution
- **Competitive Positioning:** Fair commission (85% take-home)

## Marketing Strategy
- **Target Audience:** Providers evaluating marketplace trustworthiness
- **Primary Message:** "Fair, transparent marketplace with guaranteed payments"
- **Emotional Appeal:** Trust (guaranteed payments) + Confidence (fair commission)
- **Copy Tone:** Transparent, fair, trust-building

## Variants
- **Default**: Full marketplace explanation with all features
- **PricingFocus**: Emphasize pricing model and commission structure
- **ReputationFocus**: Lead with rating system and trust signals

## Examples
\`\`\`tsx
import { ProvidersMarketplace } from '@rbee/ui/organisms/Providers/ProvidersMarketplace'

// Simple usage - no props needed
<ProvidersMarketplace />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- ProvidersEarnings (calculator)
- ProvidersSecurity (trust building)
- ProvidersFeatures (control features)

## Accessibility
- **Semantic HTML**: Proper heading hierarchy and landmark regions
- **Icon Labels**: Icons have aria-hidden="true"
- **Keyboard Navigation**: All interactive elements are keyboard accessible
- **Color Contrast**: Meets WCAG AA standards
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof ProvidersMarketplace>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersMarketplace as used on /gpu-providers page.
 * Shows full marketplace explanation with features and commission.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing pricing model and commission structure.
 * Focuses on the 85% take-home rate and transparent breakdown.
 */
export const PricingFocus: Story = {}

/**
 * Variant emphasizing trust and reputation system.
 * Leads with rating system, guaranteed payments, and dispute resolution.
 */
export const ReputationFocus: Story = {}
