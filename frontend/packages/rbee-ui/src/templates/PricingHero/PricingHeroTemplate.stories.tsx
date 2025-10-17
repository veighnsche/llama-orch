import { pricingHeroProps } from '@rbee/ui/pages/PricingPage'
import type { Meta, StoryObj } from '@storybook/react'
import { PricingHeroTemplate } from './PricingHeroTemplate'

const meta = {
  title: 'Templates/PricingHeroTemplate',
  component: PricingHeroTemplate,
  parameters: { layout: 'fullscreen' },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingHeroTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnPricingHero - pricingHeroProps
 * @tags pricing, hero, plans
 * 
 * PricingHeroTemplate as used on the Pricing page
 * - "Start Free. Scale When Ready." headline
 * - Honest Pricing badge
 * - Four assurance items with Sparkles icons
 * - PricingScaleVisual illustration
 */
export const OnPricingHero: Story = {
  args: pricingHeroProps,
}
