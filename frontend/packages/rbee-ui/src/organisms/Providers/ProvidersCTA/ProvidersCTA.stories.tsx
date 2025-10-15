import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersCTA } from './ProvidersCTA'

const meta = {
  title: 'Organisms/Providers/ProvidersCTA',
  component: ProvidersCTA,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The ProvidersCTA section (also exported as CTASectionProviders) is the final call-to-action on the GPU Providers page. It features a centered layout with headline, CTAs, and reassurance stats (<15 minutes setup, 15% platform fee, €25 minimum weekly payouts). Background includes decorative GPU earnings image.

## Two-Sided Marketplace Strategy

### Provider Conversion Strategy
- **Headline:** "Turn Idle GPUs Into Weekly Payouts" - direct benefit statement
- **Social Proof:** "Join 500+ providers monetizing spare GPU time"
- **Primary CTA:** "Start Earning Now" - immediate action
- **Secondary CTA:** "View Docs" - for those needing more info
- **Reassurance:** Setup time, commission, payout details

### Friction Reduction
- **Setup Time:** "<15 minutes" - quick time-to-value
- **Commission:** "15% platform fee, You keep 85%" - transparent pricing
- **Payouts:** "€25 minimum, Weekly payouts" - clear payout terms
- **Disclaimer:** "Data from verified providers; earnings vary" - realistic expectations

## Composition
This organism contains:
- **Background Image**: Decorative GPU earnings visual (right edge)
- **Badge**: "Start earning today" with Zap icon
- **Headline**: "Turn Idle GPUs Into Weekly Payouts"
- **Subtitle**: "Join 500+ providers monetizing spare GPU time on the rbee marketplace."
- **CTAs**:
  - Primary: "Start Earning Now" (with ArrowRight icon)
  - Secondary: "View Docs"
- **Micro-credibility**: "Data from verified providers; earnings vary"
- **Reassurance Bar (StatsGrid)**:
  - <15 minutes setup time (Clock icon)
  - 15% platform fee, You keep 85% (Shield icon)
  - €25 minimum, Weekly payouts (Wallet icon)

## When to Use
- As the final section on the GPU Providers page
- To drive provider sign-ups after building trust
- To provide clear next steps with reassurance

## Content Requirements
- **Headline:** Direct benefit statement
- **CTAs:** Clear primary and secondary actions
- **Reassurance Stats:** Setup time, commission, payout details
- **Tone:** Urgent (but not pushy), clear, reassuring

## Marketing Strategy
- **Target Audience:** Providers ready to sign up after reading page
- **Primary Message:** "Start earning now - it's quick and transparent"
- **Emotional Appeal:** Urgency (start today) + Confidence (clear terms)
- **CTAs:** 
  - Primary: "Start Earning Now" - immediate conversion
  - Secondary: "View Docs" - for cautious providers
- **Copy Tone:** Urgent, clear, reassuring

## Variants
- **Default**: Full CTA with all reassurance stats
- **EarningsFocus**: Emphasize income potential and weekly payouts
- **EasySetupFocus**: Lead with "<15 minutes setup time"

## Examples
\`\`\`tsx
import { ProvidersCTA } from '@rbee/ui/organisms/Providers/ProvidersCTA'

// Simple usage - no props needed
<ProvidersCTA />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers) - final section

## Related Components
- Button (for CTAs)
- StatsGrid (for reassurance stats)
- ProvidersHero (opening CTA)

## Accessibility
- **Semantic HTML**: Proper heading hierarchy with aria-labelledby
- **ARIA Labels**: CTAs have descriptive aria-label attributes
- **Icon Labels**: Icons have aria-hidden="true"
- **Keyboard Navigation**: All CTAs are keyboard accessible
- **Focus States**: Visible focus indicators on interactive elements
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersCTA>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersCTA as used on /gpu-providers page.
 * Shows full CTA with all reassurance stats.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing earnings and weekly payouts.
 * Highlights income potential and payout frequency.
 */
export const EarningsFocus: Story = {}

/**
 * Variant emphasizing easy setup.
 * Leads with "<15 minutes setup time" messaging.
 */
export const EasySetupFocus: Story = {}
