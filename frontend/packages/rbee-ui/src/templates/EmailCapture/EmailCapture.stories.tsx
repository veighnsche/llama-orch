import { developersEmailCaptureProps } from '@rbee/ui/pages/DevelopersPage'
import { enterpriseEmailCaptureProps } from '@rbee/ui/pages/EnterprisePage'
import { featuresEmailCaptureProps } from '@rbee/ui/pages/FeaturesPage'
import { emailCaptureProps } from '@rbee/ui/pages/HomePage'
import { pricingEmailCaptureProps } from '@rbee/ui/pages/PricingPage'
import { useCasesEmailCaptureProps } from '@rbee/ui/pages/UseCasesPage'
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
  args: emailCaptureProps,
}

/**
 * EmailCapture as used on the Features page
 * - Focus on feature updates
 * - Performance improvements messaging
 * - Community highlights
 */
export const OnFeaturesPage: Story = {
  args: featuresEmailCaptureProps,
}

/**
 * EmailCapture as used on the Use Cases page
 * - Focus on use cases and best practices
 * - Simpler messaging without badge
 */
export const OnUseCasesPage: Story = {
  args: useCasesEmailCaptureProps,
}

/**
 * EmailCapture as used on the Pricing page
 * - Focus on pricing updates
 * - Plans and billing messaging
 * - Discount notifications
 */
export const OnPricingPage: Story = {
  args: pricingEmailCaptureProps,
}

/**
 * EmailCapture as used on the Developers page
 * - Developer-focused messaging
 * - Build AI tools without vendor lock-in
 * - GitHub and Discord community links
 */
export const OnDevelopersPage: Story = {
  args: developersEmailCaptureProps,
}

/**
 * EmailCapture as used on the Enterprise page
 * - Enterprise-focused messaging
 * - GDPR-compliant AI infrastructure
 * - Compliance and security focus
 */
export const OnEnterprisePage: Story = {
  args: enterpriseEmailCaptureProps,
}
