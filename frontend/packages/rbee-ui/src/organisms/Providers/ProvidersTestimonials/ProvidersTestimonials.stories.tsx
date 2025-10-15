import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersTestimonials } from './ProvidersTestimonials'

const meta = {
	title: 'Organisms/Providers/ProvidersTestimonials',
	component: ProvidersTestimonials,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProvidersTestimonials section (also exported as SocialProofSection) showcases provider testimonials via a carousel rail and displays marketplace statistics (500+ Active Providers, 2,000+ GPUs Earning, €180K+ Paid to Providers, 4.8/5 Average Rating). It provides social proof and credibility for the provider marketplace.

## Two-Sided Marketplace Strategy

### Social Proof Elements
- **Provider Count:** "500+ Active Providers" - marketplace is active
- **GPU Count:** "2,000+ GPUs Earning" - scale and liquidity
- **Total Paid:** "€180K+ Paid to Providers" - proof of actual payouts
- **Rating:** "4.8/5 Average Rating" - provider satisfaction

### Credibility Strategy
- **Real Testimonials:** Carousel of provider stories (filtered by sectorFilter="provider")
- **Verified Data:** "Data from verified providers on rbee"
- **Transparency:** "Payouts vary by GPU, uptime, and demand"
- **Social Proof:** Large numbers (500+, 2,000+, €180K+) build confidence

### Testimonial Themes
- **Earnings Focus:** "Earning ~€150/mo while I'm at work or asleep"
- **Ease Focus:** "Setup was easier than I expected"
- **ROI Focus:** "Better than mining Bitcoin"

## Composition
This organism contains:
- **Header**:
  - Kicker: "Provider Stories"
  - Title: "What Real Providers Are Earning"
  - Subtitle: "GPU owners on the rbee marketplace turn idle time into steady payouts"
  - Disclaimer: "Data from verified providers; payouts vary"
- **TestimonialsRail**: Carousel with sectorFilter="provider"
- **StatsGrid** (4-column grid):
  1. 500+ Active Providers (Users icon)
  2. 2,000+ GPUs Earning (Cpu icon)
  3. €180K+ Paid to Providers (TrendingUp icon)
  4. 4.8/5 Average Rating (Star icon)

## When to Use
- On the GPU Providers page near the end (before CTA)
- To provide social proof and credibility
- To showcase real provider earnings and satisfaction

## Content Requirements
- **Testimonials:** Real provider stories (via TestimonialsRail)
- **Stats:** Marketplace metrics (providers, GPUs, payouts, rating)
- **Disclaimer:** "Data from verified providers; payouts vary"
- **Tone:** Authentic, credible, earnings-focused

## Marketing Strategy
- **Target Audience:** Potential providers evaluating credibility
- **Primary Message:** "Real providers are earning real money on rbee"
- **Emotional Appeal:** Validation (others are succeeding) + Confidence (large scale)
- **Copy Tone:** Authentic, credible, social proof

## Variants
- **Default**: All testimonials and stats
- **EarningsTestimonials**: Filter testimonials to earnings-focused stories
- **EaseTestimonials**: Filter testimonials to simplicity-focused stories

## Examples
\`\`\`tsx
import { ProvidersTestimonials } from '@rbee/ui/organisms/Providers/ProvidersTestimonials'

// Simple usage - no props needed
<ProvidersTestimonials />
\`\`\`

## Used In
- GPU Providers page (/gpu-providers)

## Related Components
- TestimonialsRail (carousel component)
- StatsGrid (statistics display)
- ProvidersUseCases (persona stories)

## Accessibility
- **Semantic HTML**: Proper heading hierarchy with aria-labelledby
- **Icon Labels**: Icons have aria-hidden="true"
- **Carousel**: TestimonialsRail has proper ARIA roles
- **Screen Readers**: Stats and testimonials are properly announced
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof ProvidersTestimonials>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default ProvidersTestimonials as used on /gpu-providers page.
 * Shows all provider testimonials and marketplace stats.
 */
export const ProvidersPageDefault: Story = {}

/**
 * Variant emphasizing earnings testimonials.
 * Filters to stories focused on income and payouts.
 */
export const EarningsTestimonials: Story = {}

/**
 * Variant emphasizing ease-of-use testimonials.
 * Filters to stories focused on simplicity and setup.
 */
export const EaseTestimonials: Story = {}
