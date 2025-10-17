import { TemplateContainer } from '@rbee/ui/molecules'
import { developersEmailCaptureContainerProps, developersEmailCaptureProps } from '@rbee/ui/pages/DevelopersPage'
import { enterpriseEmailCaptureContainerProps, enterpriseEmailCaptureProps } from '@rbee/ui/pages/EnterprisePage'
import { featuresEmailCaptureContainerProps, featuresEmailCaptureProps } from '@rbee/ui/pages/FeaturesPage'
import { emailCaptureContainerProps, emailCaptureProps } from '@rbee/ui/pages/HomePage'
import { pricingEmailCaptureContainerProps, pricingEmailCaptureProps } from '@rbee/ui/pages/PricingPage'
import { useCasesEmailCaptureContainerProps, useCasesEmailCaptureProps } from '@rbee/ui/pages/UseCasesPage'
import type { Meta, StoryObj } from '@storybook/react'
import { EmailCapture } from './EmailCapture'

const meta = {
  title: 'Templates/EmailCapture',
  component: EmailCapture,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EmailCapture>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EmailCapture as used on the Home page
 * - Badge with development status and pulse
 * - Community-focused messaging
 * - GitHub and Discord links
 * - Homelab bee illustration
 */
export const OnHomePage: Story = {
  render: (args) => (
    <TemplateContainer {...emailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: emailCaptureProps,
}

/**
 * EmailCapture as used on the Features page
 * - Focus on feature updates
 * - Performance improvements messaging
 * - Community highlights
 */
export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...featuresEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: featuresEmailCaptureProps,
}

/**
 * EmailCapture as used on the Use Cases page
 * - Focus on use cases and best practices
 * - Simpler messaging without badge
 */
export const OnUseCasesPage: Story = {
  render: (args) => (
    <TemplateContainer {...useCasesEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: useCasesEmailCaptureProps,
}

/**
 * EmailCapture as used on the Pricing page
 * - Focus on pricing updates
 * - Plans and billing messaging
 * - Discount notifications
 */
export const OnPricingPage: Story = {
  render: (args) => (
    <TemplateContainer {...pricingEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: pricingEmailCaptureProps,
}

/**
 * EmailCapture as used on the Developers page
 * - Developer-focused messaging
 * - Build AI tools without vendor lock-in
 * - GitHub and Discord community links
 */
export const OnDevelopersPage: Story = {
  render: (args) => (
    <TemplateContainer {...developersEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: developersEmailCaptureProps,
}

/**
 * EmailCapture as used on the Enterprise page
 * - Enterprise-focused messaging
 * - GDPR-compliant AI infrastructure
 * - Compliance and security focus
 */
export const OnEnterprisePage: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseEmailCaptureContainerProps}>
      <EmailCapture {...args} />
    </TemplateContainer>
  ),
  args: enterpriseEmailCaptureProps,
}
