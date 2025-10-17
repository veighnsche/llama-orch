import { howItWorksContainerProps as developersHowItWorksContainerProps, howItWorksProps as developersHowItWorksProps } from '@rbee/ui/pages/DevelopersPage'
import { howItWorksContainerProps, howItWorksProps } from '@rbee/ui/pages/HomePage'
import { providersHowItWorksContainerProps, providersHowItWorksProps } from '@rbee/ui/pages/ProvidersPage'
import { TemplateContainer } from '@rbee/ui/molecules'
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
  render: (args) => (
    <TemplateContainer {...howItWorksContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: howItWorksProps,
}

/**
 * HowItWorks as used on the Developers page
 * - Four-step installation guide
 * - Developer-focused messaging
 * - Terminal blocks and code examples
 */
export const OnDevelopersPage: Story = {
  render: (args) => (
    <TemplateContainer {...developersHowItWorksContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
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
  render: (args) => (
    <TemplateContainer {...providersHowItWorksContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: providersHowItWorksProps,
}
