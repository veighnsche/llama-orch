import type { Meta, StoryObj } from '@storybook/react'
import { pricingHeroProps } from '@rbee/ui/pages'
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
 * PricingHeroTemplate as used on the Pricing page
 * - "Start Free. Scale When Ready." headline
 * - Honest Pricing badge
 * - Four assurance items with Sparkles icons
 * - PricingScaleVisual illustration
 */
export const OnPricingPage: Story = {
  args: pricingHeroProps,
}
