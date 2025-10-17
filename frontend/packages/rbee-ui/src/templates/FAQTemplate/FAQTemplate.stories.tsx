import { faqTemplateContainerProps, faqTemplateProps } from '@rbee/ui/pages/HomePage'
import { pricingFaqContainerProps, pricingFaqProps } from '@rbee/ui/pages/PricingPage'
import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { FAQTemplate } from './FAQTemplate'

const meta = {
  title: 'Templates/FAQTemplate',
  component: FAQTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof FAQTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * FAQTemplate as used on the Home page
 * - Badge: "Support • Self-hosted AI"
 * - 8 general questions across 6 categories
 * - Categories: Setup, Models, Performance, Marketplace, Security, Production
 * - Includes support card with beehive illustration
 * - Links to GitHub discussions, setup guide, email support
 */
export const OnHomePage: Story = {
  render: (args) => (
    <TemplateContainer {...faqTemplateContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: faqTemplateProps,
}

/**
 * FAQTemplate as used on the Pricing page
 * - Badge: "Pricing • Plans & Billing"
 * - 6 pricing-specific questions across 4 categories
 * - Categories: Licensing, Plans, Billing, Trials
 * - No support card (pricing-focused)
 * - Questions about free tier, upgrades, payment methods, discounts, trials
 */
export const OnPricingPage: Story = {
  render: (args) => (
    <TemplateContainer {...pricingFaqContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: pricingFaqProps,
}
