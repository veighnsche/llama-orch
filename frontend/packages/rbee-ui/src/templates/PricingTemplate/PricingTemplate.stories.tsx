import type { Meta, StoryObj } from '@storybook/react'
import {
  pricingTemplateProps,
  developersPricingTemplateProps,
} from '@rbee/ui/pages'
import { pricingTemplateProps as homePricingTemplateProps } from '@rbee/ui/pages/HomePage'
import { PricingTemplate } from './PricingTemplate'

const meta = {
  title: 'Templates/PricingTemplate',
  component: PricingTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * PricingTemplate as used on the Home page
 * - Kicker badges (Open source, OpenAI-compatible, Multi-GPU, No feature gates)
 * - Editorial image below tiers
 * - Full featured variant
 */
export const OnHomePage: Story = {
  args: homePricingTemplateProps,
}

/**
 * PricingTemplate as used on the Pricing page
 * - No kicker badges
 * - No editorial image
 * - Pricing-focused footer text
 */
export const OnPricingPage: Story = {
  args: pricingTemplateProps,
}

/**
 * PricingTemplate as used on the Developers page
 * - No kicker badges
 * - No editorial image
 * - Developer-focused footer text
 */
export const OnDevelopersPage: Story = {
  args: developersPricingTemplateProps,
}
