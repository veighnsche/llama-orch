import type { Meta, StoryObj } from '@storybook/react'
import { PricingHero } from './PricingHero'

// Created by: TEAM-004

const meta = {
  title: 'Organisms/Pricing/PricingHero',
  component: PricingHero,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component: `
## Overview
The PricingHero is the primary landing section for the Pricing page, emphasizing transparent pricing, no feature gates, and "Start Free, Scale When Ready" messaging. Features a split layout with messaging on the left and a pricing scale visual on the right.

## Composition
This organism contains:
- **Badge**: "Honest Pricing" eyebrow
- **H1 Headline**: "Start Free. Scale When Ready." with primary color accent
- **Support Copy**: "Every tier ships the full rbee orchestrator—no feature gates, no artificial limits"
- **CTA Buttons**: "View Plans" (primary) and "Talk to Sales" (secondary)
- **Assurance Checkmarks**: 4 key benefits (Full orchestrator on every tier, No feature gates or limits, OpenAI-compatible API, Cancel anytime)
- **PricingScaleVisual**: Illustration showing pricing scales from single-GPU homelab to multi-node server setups

## When to Use
- As the first section on the /pricing page
- To immediately communicate pricing philosophy
- To differentiate from competitors with feature gates
- To drive users to view plans or contact sales

## Content Requirements
- **Headline**: Clear pricing philosophy (transparent, no gates)
- **Support Copy**: Emphasize no feature gates, full orchestrator on every tier
- **Assurance Checkmarks**: Key benefits that reduce buyer anxiety
- **CTAs**: View plans (self-serve) and talk to sales (enterprise)

## Variants
- **Default**: Full hero with all elements
- **Value First**: Emphasize value over price
- **Transparency First**: Emphasize pricing transparency

## Examples
\`\`\`tsx
import { PricingHero } from '@rbee/ui/organisms/Pricing/PricingHero'

// Simple usage - no props needed
<PricingHero />
\`\`\`

## Used In
- Pricing page (/pricing)

## Related Components
- Badge
- Button
- PricingScaleVisual

## Accessibility
- **Keyboard Navigation**: All buttons are keyboard accessible
- **ARIA Labels**: PricingScaleVisual has descriptive aria-label
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Focus States**: Visible focus indicators
- **Color Contrast**: Meets WCAG AA standards

## Pricing Strategy

### Pricing Philosophy
- **Transparent**: No hidden fees, no surprise charges
- **No Feature Gates**: Full orchestrator on every tier (not "lite" versions)
- **Start Free**: Open-source version is fully functional (not a trial)
- **Scale When Ready**: Pay only when you need enterprise features (SLAs, support, white-label)

### Pricing Psychology

**Anchor Pricing**
- **Reference Price**: Cloud APIs (OpenAI, Anthropic) at $0.03-$0.10 per 1K tokens
- **rbee Positioning**: Fixed infrastructure cost (predictable) vs. variable per-token cost (unpredictable)
- **Value Demonstration**: "Save 60-80% at scale vs. cloud APIs"

**Pricing Transparency**
- **No Hidden Fees**: All costs upfront (infrastructure, support, licenses)
- **No Surprise Charges**: Fixed monthly/annual pricing (not per-token)
- **No Feature Gates**: Full orchestrator on every tier (not "lite" versions)
- **Cancel Anytime**: No long-term contracts (except enterprise with SLAs)

**Objection Handling**
- **"Too expensive?"**: Show cost comparison vs. cloud APIs at scale
- **"Why not free forever?"**: Explain enterprise features (SLAs, support, white-label) require investment
- **"What if I outgrow free tier?"**: Emphasize smooth upgrade path, no migration needed

### Tier Structure

**Free/OSS (Home/Lab)**
- **Price**: $0 (open-source)
- **Target**: Solo developers, hobbyists, homelab enthusiasts
- **Features**: Full orchestrator, unlimited GPUs, OpenAI-compatible API, CLI access
- **Limitations**: No web UI, no team collaboration, community support only
- **Upgrade Trigger**: Need web UI, team collaboration, or priority support

**Pro (Team)**
- **Price**: TBD (monthly/annual)
- **Target**: Small teams (2-10 people)
- **Features**: Everything in Free + Web UI, team collaboration, priority email support
- **Limitations**: No SLA, no white-label, no professional services
- **Upgrade Trigger**: Need SLA, white-label, or professional services

**Enterprise**
- **Price**: Custom (contact sales)
- **Target**: Large enterprises (50-5000+ employees)
- **Features**: Everything in Pro + 99.9% uptime SLA, 24/7 support, dedicated account manager, white-label, professional services, multi-region support
- **Limitations**: None (all features included)
- **Upgrade Trigger**: N/A (top tier)

### Conversion Strategy
- **Primary CTA**: "View Plans" (self-serve, see pricing tiers)
- **Secondary CTA**: "Talk to Sales" (enterprise, custom pricing)
- **Lead qualification**: Capture use case, team size, infrastructure details
- **Proof points**: No feature gates, full orchestrator on every tier, OpenAI-compatible API, cancel anytime
				`,
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingHero>

export default meta
type Story = StoryObj<typeof meta>

export const PricingPageDefault: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Default pricing hero as seen on /pricing page. Emphasizes "Start Free. Scale When Ready." messaging with transparent pricing philosophy: full orchestrator on every tier, no feature gates, no artificial limits, OpenAI-compatible API, and cancel anytime. Includes PricingScaleVisual illustration showing pricing scales from single-GPU homelab to multi-node server setups.',
      },
    },
  },
}

export const ValueFirst: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing value over price. This version would lead with: "Save 60-80% at scale vs. cloud APIs", "Predictable costs (no surprise bills)", "Full orchestrator on every tier (no feature gates)", and "OpenAI-compatible API (drop-in replacement)". Includes cost comparison calculator showing savings vs. OpenAI/Anthropic at different usage levels. Ideal for cost-conscious buyers or those burned by cloud API bills.',
      },
    },
  },
}

export const TransparencyFirst: Story = {
  parameters: {
    docs: {
      description: {
        story:
          'Variant emphasizing pricing transparency. This version would lead with: "No hidden fees", "No surprise charges", "No feature gates (full orchestrator on every tier)", "No long-term contracts (cancel anytime)", and "Open-source foundation (no vendor lock-in)". Includes detailed pricing breakdown showing all costs upfront (infrastructure, support, licenses). Ideal for buyers skeptical of SaaS pricing or those burned by hidden fees.',
      },
    },
  },
}
