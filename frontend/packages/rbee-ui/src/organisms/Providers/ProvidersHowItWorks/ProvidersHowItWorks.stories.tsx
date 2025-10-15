import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersHowItWorks } from './ProvidersHowItWorks'

const meta = {
  title: 'Organisms/Providers/ProvidersHowItWorks',
  component: ProvidersHowItWorks,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The ProvidersHowItWorks section breaks down the provider onboarding flow into 4 simple steps. It uses the shared StepsSection component with provider-specific content focused on ease of setup and quick time-to-earnings.

## Two-Sided Marketplace Strategy

### Provider Onboarding Flow
1. **Install rbee:** One-command install (curl | sh)
2. **Configure Settings:** Set pricing, availability, limits via web dashboard
3. **Join Marketplace:** GPUs automatically listed, discoverable by developers
4. **Get Paid:** Real-time earnings tracking, weekly payouts (€25 minimum)

### Friction Reduction
- **Technical Friction:** "One command install" + "No technical expertise required"
- **Configuration Friction:** "Intuitive web dashboard" + "~15 minutes total"
- **Trust Friction:** "Real-time earnings tracking" + "Automatic payouts"

### Time-to-Value
- **Average Setup Time:** 12 minutes (shown in component)
- **Time to First Earnings:** As soon as first job runs (could be same day)

## Composition
This organism contains:
- **Kicker**: "How rbee Works"
- **Title**: "Start Earning in 4 Simple Steps"
- **Subtitle**: "No technical expertise required. Most providers finish in ~15 minutes."
- **Steps**: Four sequential steps with icons, titles, bodies, and supporting content
  1. Install rbee (with code snippet)
  2. Configure Settings (with checklist)
  3. Join Marketplace (with success note)
  4. Get Paid (with payout stats)
- **Average Time**: "12 minutes" badge

## When to Use
- On the GPU Providers page to explain onboarding
- To reduce perceived complexity and friction
- To show clear path from signup to earnings

## Content Requirements
- **Steps**: Must be sequential and actionable
- **Time Estimate**: Realistic average setup time
- **Supporting Content**: Code snippets, checklists, stats to build confidence
- **Tone**: Encouraging, simple, friction-reducing

## Marketing Strategy
- **Target Audience:** GPU owners concerned about complexity
- **Primary Message:** "You can start earning in ~15 minutes"
- **Emotional Appeal:** Confidence (it's easy) + Excitement (quick earnings)
- **Copy Tone:** Encouraging, simple, step-by-step

## Variants
- **Default**: Full 4-step flow with all supporting content
- **SimplifiedFlow**: Emphasize ease and speed (remove technical details)
- **DetailedFlow**: Add more technical depth for advanced users

## Examples
\`\`\`tsx
import { ProvidersHowItWorks } from '@rbee/ui/organisms/Providers/ProvidersHowItWorks'

// Simple usage - no props needed
<ProvidersHowItWorks />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- StepsSection (shared base component)
- ProvidersSolution
- ProvidersFeatures

## Accessibility
- **Semantic HTML**: Ordered list structure for steps
- **Icon Labels**: Icons have aria-hidden="true"
- **Keyboard Navigation**: All interactive elements are keyboard accessible
- **Screen Readers**: Step numbers and titles are properly announced
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersHowItWorks>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersHowItWorks as used on /gpu-providers page.
 * Shows full 4-step onboarding flow with supporting content.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing simplicity and speed.
 * Focuses on "~15 minutes" and "No technical expertise required."
 */
export const SimplifiedFlow: Story = {}

/**
 * Variant with more technical depth.
 * Adds configuration details for advanced users.
 */
export const DetailedFlow: Story = {}
