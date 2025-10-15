import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersUseCases } from './ProvidersUseCases'

const meta = {
  title: 'Organisms/Providers/ProvidersUseCases',
  component: ProvidersUseCases,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The ProvidersUseCases section showcases four provider personas (Gaming PC Owners, Homelab Enthusiasts, Former Crypto Miners, Workstation Owners) with real earnings data and testimonials. It uses a 2-column grid layout with case cards featuring icons, quotes, and earnings breakdowns.

## Two-Sided Marketplace Strategy

### Provider Personas
1. **Gaming PC Owners** (Most common): RTX 4080-4090, 16-20h/day availability, €120-180/mo
2. **Homelab Enthusiasts** (Multiple GPUs): 3-6 GPUs, 20-24h/day availability, €300-600/mo
3. **Former Crypto Miners** (Repurposed rigs): 6-12 GPUs, 24h/day availability, €600-1,200/mo
4. **Workstation Owners** (Professional GPUs): RTX 4070-4080, 12-16h/day availability, €80-140/mo

### Persona Priorities
- **Gamers:** Earn while idle (at work/asleep), no impact on gaming performance
- **Homelabbers:** Maximize ROI on existing hardware, cover electricity costs + profit
- **Miners:** Better margins than crypto mining post-PoS, repurpose existing infrastructure
- **Workstation Owners:** Monetize idle time between rendering/design work

### Earnings Expectations
- **Realistic Ranges:** €80-1,200/mo depending on GPU count and availability
- **Transparency:** Each persona shows typical GPU, availability hours, and monthly earnings
- **Social Proof:** Testimonial quotes from each persona type

## Composition
This organism contains:
- **Kicker**: "Real Providers, Real Earnings"
- **Title**: "Who's Earning with rbee?"
- **Subtitle**: "From gamers to homelab builders, anyone with a spare GPU can turn idle time into income."
- **Case Cards Grid**: 2-column grid with 4 case cards
  - Icon + Title + Subtitle
  - Testimonial quote
  - Facts (GPU type, availability, monthly earnings)
  - Persona illustration
- **CTAs**: "Start Earning" (primary) and "Estimate My Payout" (secondary)

## When to Use
- On the GPU Providers page to showcase provider diversity
- To provide social proof and earnings validation
- To help potential providers identify with a persona

## Content Requirements
- **Personas**: Must represent diverse provider types
- **Earnings**: Realistic ranges with context (GPU type, availability)
- **Testimonials**: First-person quotes that feel authentic
- **Illustrations**: Visual representation of each persona's setup

## Marketing Strategy
- **Target Audience:** Potential GPU providers evaluating opportunity
- **Primary Message:** "People like you are already earning with rbee"
- **Emotional Appeal:** Identification (persona match) + Validation (real earnings)
- **CTAs:** 
  - Primary: "Start Earning" - join the community
  - Secondary: "Estimate My Payout" - personalized calculation
- **Copy Tone:** Authentic, relatable, earnings-focused

## Variants
- **Default**: All four personas with earnings data
- **GamerFocus**: Single persona deep dive (gaming PC owner)
- **ComparativeROI**: Emphasize earnings vs. mining/cloud alternatives

## Examples
\`\`\`tsx
import { ProvidersUseCases } from '@rbee/ui/organisms/Providers/ProvidersUseCases'

// Simple usage - no props needed
<ProvidersUseCases />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- UseCasesSection (shared base component)
- ProvidersFeatures
- ProvidersEarnings

## Accessibility
- **Semantic HTML**: Proper heading hierarchy and card structure
- **Icon Labels**: Icons have aria-hidden="true"
- **Keyboard Navigation**: All interactive elements are keyboard accessible
- **Screen Readers**: Testimonial quotes and earnings are properly announced
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersUseCases>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersUseCases as used on /gpu-providers page.
 * Shows all four provider personas with earnings data.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant focusing on a single persona (gaming PC owner).
 * Deep dive into the most common provider type.
 */
export const GamerFocus: Story = {}

/**
 * Variant emphasizing comparative ROI.
 * Highlights earnings vs. crypto mining and cloud rental alternatives.
 */
export const ComparativeROI: Story = {}
