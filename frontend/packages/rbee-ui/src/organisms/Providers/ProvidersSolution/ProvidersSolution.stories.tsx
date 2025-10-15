import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersSolution } from './ProvidersSolution'

const meta = {
	title: 'Organisms/Providers/ProvidersSolution',
	component: ProvidersSolution,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProvidersSolution section presents the rbee marketplace as the solution to idle GPU problems. It features four key benefits, a 4-step onboarding flow, and an earnings table showing realistic monthly income by GPU model.

## Two-Sided Marketplace Strategy

### Provider Value Proposition
- **Passive Income:** €50-200/month per GPU
- **Full Control:** Set prices, availability, usage limits
- **Secure & Private:** Sandboxed jobs, no file access
- **Easy Setup:** ~10 minutes, no expertise required

### Trust Building
- **Security:** "Sandboxed jobs. No access to your files."
- **Control:** "Set prices, availability windows, and usage limits."
- **Transparency:** Earnings table with conservative estimates and disclaimer

### Risk Mitigation
- **Technical Risk:** "Easy Setup" + "No expertise required"
- **Security Risk:** "Sandboxed jobs" + "No access to your files"
- **Financial Risk:** Conservative earnings estimates with disclaimer

## Composition
This organism contains:
- **Kicker**: "How rbee Works"
- **Title**: "Turn Idle GPUs Into Reliable Monthly Income"
- **Subtitle**: Marketplace explanation
- **Features Grid**: Four key benefits (Passive Income, Full Control, Secure & Private, Easy Setup)
- **Steps**: 4-step onboarding flow (Install → Configure → Join Marketplace → Get Paid)
- **Earnings Table**: Three GPU models with monthly earnings at 80% utilization
  - RTX 4090: €180/mo
  - RTX 4080: €140/mo
  - RTX 3080: €90/mo
- **Disclaimer**: "Actuals vary with demand, pricing, and availability. These are conservative estimates."
- **CTAs**: "Start Earning" (primary) and "Estimate My Payout" (secondary)

## When to Use
- On the GPU Providers page after the problem section
- To present the marketplace solution and earning potential
- To build trust through transparency and control messaging

## Content Requirements
- **Features**: Must address provider objections (security, control, complexity)
- **Steps**: Clear onboarding flow (4 steps max)
- **Earnings Table**: Realistic estimates with disclaimer
- **CTAs**: Link to onboarding and earnings calculator

## Marketing Strategy
- **Target Audience:** GPU owners ready to monetize
- **Primary Message:** "rbee turns idle GPUs into reliable monthly income"
- **Emotional Appeal:** Empowerment (control) + Freedom (passive income)
- **CTAs:** 
  - Primary: "Start Earning" - begin onboarding
  - Secondary: "Estimate My Payout" - personalized calculator
- **Copy Tone:** Empowering, transparent, trust-building

## Variants
- **Default**: Full solution with all features, steps, and earnings table
- **EarningsCalculator**: Emphasize earnings table and calculator CTA
- **SecurityFirst**: Lead with security/sandboxing messaging

## Examples
\`\`\`tsx
import { ProvidersSolution } from '@rbee/ui/organisms/Providers/ProvidersSolution'

// Simple usage - no props needed
<ProvidersSolution />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- SolutionSection (shared base component)
- ProvidersProblem
- ProvidersHowItWorks

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
} satisfies Meta<typeof ProvidersSolution>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersSolution as used on /gpu-providers page.
 * Shows full solution with features, steps, and earnings table.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing earnings calculator and income potential.
 * Highlights the earnings table and "Estimate My Payout" CTA.
 */
export const EarningsCalculator: Story = {}

/**
 * Variant leading with security and sandboxing.
 * Addresses FUD (fear, uncertainty, doubt) about sharing GPUs.
 */
export const SecurityFirst: Story = {}
