import { howItWorksProps as developersHowItWorksProps } from '@rbee/ui/pages/DevelopersPage'
import { howItWorksProps } from '@rbee/ui/pages/HomePage'
import { providersHowItWorksProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { HowItWorks } from './HowItWorks'

const meta = {
  title: 'Templates/HowItWorks',
  component: HowItWorks,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof HowItWorks>

export default meta
type Story = StoryObj<typeof meta>

/**
 * HowItWorks as used on the Home page
 * - Four-step installation guide
 * - Terminal and code blocks
 * - Focus on quick setup across all hardware
 */
export const OnHomePage: Story = {
  args: howItWorksProps,
}

/**
 * HowItWorks as used on the Developers page
 * - Four-step installation guide
 * - Developer-focused messaging
 * - Terminal blocks and code examples
 */
export const OnDevelopersPage: Story = {
  args: developersHowItWorksProps,
}

/**
 * HowItWorks as used on the Providers page
 * - Four-step guide to start earning
 * - Terminal installation
 * - Configuration and marketplace setup
 * - Payout tracking
 */
export const OnProvidersPage: Story = {
  args: providersHowItWorksProps,
}
