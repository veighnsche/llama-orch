import { TemplateContainer } from '@rbee/ui/molecules'
import { pricingTemplateContainerProps as homePricingTemplateContainerProps, pricingTemplateProps as homePricingTemplateProps,  } from '@rbee/ui/pages/HomePage'
import { pricingTemplateContainerProps, pricingTemplateProps } from '@rbee/ui/pages/PricingPage'
import { educationCourseLevelsContainerProps, educationCourseLevelsProps } from '@rbee/ui/pages/EducationPage'
import type { Meta, StoryObj } from '@storybook/react'
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
 * OnHomePricing - pricingTemplateProps
 * @tags home, pricing, tiers
 *
 * PricingTemplate as used on the Home page
 * - Kicker badges (Open source, OpenAI-compatible, Multi-GPU, No feature gates)
 * - Editorial image below tiers
 * - Full featured variant
 */
export const OnHomePricing: Story = {
  render: (args) => (
    <TemplateContainer {...homePricingTemplateContainerProps}>
      <PricingTemplate {...args} />
    </TemplateContainer>
  ),
  args: homePricingTemplateProps,
}

/**
 * OnHomePricing - pricingTemplateProps
 * @tags home, pricing, tiers
 *
 * PricingTemplate as used on the Pricing page
 * - No kicker badges
 * - No editorial image
 * - Pricing-focused footer text
 */
export const OnPricingPage: Story = {
  render: (args) => (
    <TemplateContainer {...pricingTemplateContainerProps}>
      <PricingTemplate {...args} />
    </TemplateContainer>
  ),
  args: pricingTemplateProps,
}

// Note: DevelopersPage doesn't have its own pricing template props

/**
 * PricingTemplate as used on the Education page
 * - Structured curriculum levels
 * - Beginner, Intermediate, Advanced modules
 * - Progressive learning path
 */
export const OnEducationPricing: Story = {
  render: (args) => (
    <TemplateContainer {...educationCourseLevelsContainerProps}>
      <PricingTemplate {...args} />
    </TemplateContainer>
  ),
  args: educationCourseLevelsProps,
}
