import { TemplateContainer } from '@rbee/ui/molecules'
import { communityFAQContainerProps, communityFAQProps } from '@rbee/ui/pages/CommunityPage'
import { complianceFAQContainerProps, complianceFAQProps } from '@rbee/ui/pages/CompliancePage'
import { devopsFAQContainerProps, devopsFAQProps } from '@rbee/ui/pages/DevOpsPage'
import { educationFAQContainerProps, educationFAQProps } from '@rbee/ui/pages/EducationPage'
import { faqTemplateContainerProps, faqTemplateProps } from '@rbee/ui/pages/HomePage'
import { pricingFaqContainerProps, pricingFaqProps } from '@rbee/ui/pages/PricingPage'
import { faqContainerProps, faqProps } from '@rbee/ui/pages/ResearchPage'
import { securityFAQsContainerProps, securityFAQsProps } from '@rbee/ui/pages/SecurityPage'
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
 * OnPricingFAQ - pricingFaqProps
 * @tags pricing, faq, billing
 *
 * OnHomeFAQ - faqTemplateProps
 * @tags home, faq, support
 *
 * FAQTemplate as used on the Home page
 * - Badge: "Support • Self-hosted AI"
 * - 8 general questions across 6 categories
 * - Categories: Setup, Models, Performance, Marketplace, Security, Production
 * - Includes support card with beehive illustration
 * - Links to GitHub discussions, setup guide, email support
 */
export const OnHomeFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...faqTemplateContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: faqTemplateProps,
}

/**
 * OnPricingFAQ - pricingFaqProps
 * @tags pricing, faq, billing
 *
 * OnHomeFAQ - faqTemplateProps
 * @tags home, faq, support
 *
 * FAQTemplate as used on the Pricing page
 * - Badge: "Pricing • Plans & Billing"
 * - 6 pricing-specific questions across 4 categories
 * - Categories: Licensing, Plans, Billing, Trials
 * - No support card (pricing-focused)
 * - Questions about free tier, upgrades, payment methods, discounts, trials
 */
export const OnPricingFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...pricingFaqContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: pricingFaqProps,
}

/**
 * FAQTemplate as used on the Education page
 * - Common questions about learning
 * - Prerequisites, GPU requirements
 * - Completion time
 */
export const OnEducationFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...educationFAQContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: educationFAQProps,
}

/**
 * FAQTemplate as used on the Community page
 * - Community page usage
 */
export const OnCommunityFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...communityFAQContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: communityFAQProps,
}

/**
 * FAQTemplate as used on the Compliance page
 * - Compliance page usage
 */
export const OnComplianceFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...complianceFAQContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: complianceFAQProps,
}

/**
 * FAQTemplate as used on the DevOps page
 * - DevOps page usage
 */
export const OnDevOpsFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...devopsFAQContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: devopsFAQProps,
}

/**
 * FAQTemplate as used on the Research page
 * - Research page usage
 */
export const OnResearchFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...faqContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: faqProps,
}

/**
 * FAQTemplate as used on the Security page
 * - Security page usage
 */
export const OnSecurityFAQ: Story = {
  render: (args) => (
    <TemplateContainer {...securityFAQsContainerProps}>
      <FAQTemplate {...args} />
    </TemplateContainer>
  ),
  args: securityFAQsProps,
}
