import type { Meta, StoryObj } from '@storybook/react'
import { 
  emailCaptureProps,
  featuresEmailCaptureProps,
  useCasesEmailCaptureProps,
  pricingEmailCaptureProps,
} from '@rbee/ui/pages'
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
