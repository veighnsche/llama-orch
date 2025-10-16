import type { Meta, StoryObj } from '@storybook/react'
import { PricingScaleVisual } from '@rbee/ui/icons'
import { Sparkles } from 'lucide-react'
import { PricingHeroTemplate } from './PricingHeroTemplate'

const meta = {
  title: 'Templates/PricingHeroTemplate',
  component: PricingHeroTemplate,
  parameters: { layout: 'fullscreen' },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingHeroTemplate>

export default meta
type Story = StoryObj<typeof meta>

export const OnPricingPage: Story = {
  args: {
    badgeText: 'Honest Pricing',
    heading: (
      <>
        Start Free.
        <br />
        <span className="text-primary">Scale When Ready.</span>
      </>
    ),
    description:
      'Every tier ships the full rbee orchestrator—no feature gates, no artificial limits. OpenAI-compatible API, same power on day one. Pay only when you grow.',
    primaryCta: {
      text: 'View Plans',
    },
    secondaryCta: {
      text: 'Talk to Sales',
    },
    assuranceItems: [
      { text: 'Full orchestrator on every tier', icon: Sparkles },
      { text: 'No feature gates or limits', icon: Sparkles },
      { text: 'OpenAI-compatible API', icon: Sparkles },
      { text: 'Cancel anytime', icon: Sparkles },
    ],
    visual: (
      <PricingScaleVisual
        size="100%"
        className="rounded-xl opacity-70"
        aria-label="Illustration showing rbee pricing scales from single-GPU homelab to multi-node server setups with progressive cost tiers"
      />
    ),
    visualAriaLabel:
      'Illustration showing rbee pricing scales from single-GPU homelab to multi-node server setups with progressive cost tiers',
  },
}
