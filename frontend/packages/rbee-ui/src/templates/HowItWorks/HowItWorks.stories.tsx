import { TemplateContainer } from '@rbee/ui/molecules'
import {
  howItWorksContainerProps as developersHowItWorksContainerProps,
  howItWorksProps as developersHowItWorksProps,
} from '@rbee/ui/pages/DevelopersPage'
import { howItWorksContainerProps, howItWorksProps } from '@rbee/ui/pages/HomePage'
import { providersHowItWorksContainerProps, providersHowItWorksProps } from '@rbee/ui/pages/ProvidersPage'
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
 * OnProvidersHowItWorks - providersHowItWorksProps
 * @tags providers, how-it-works, earnings
 *
 * OnDevelopersHowItWorks - howItWorksProps
 * @tags developers, how-it-works, deployment
 *
 * OnHomeHowItWorks - howItWorksProps
 * @tags home, how-it-works, steps
 *
 * HowItWorks as used on the Home page
 * - Four-step installation guide
 * - Terminal and code blocks
 * - Focus on quick setup across all hardware
 */
export const OnHomeHowItWorks: Story = {
  render: (args) => (
    <TemplateContainer {...howItWorksContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: howItWorksProps,
}

/**
 * OnProvidersHowItWorks - providersHowItWorksProps
 * @tags providers, how-it-works, earnings
 *
 * OnDevelopersHowItWorks - howItWorksProps
 * @tags developers, how-it-works, deployment
 *
 * OnHomeHowItWorks - howItWorksProps
 * @tags home, how-it-works, steps
 *
 * HowItWorks as used on the Developers page
 * - Four-step installation guide
 * - Developer-focused messaging
 * - Terminal blocks and code examples
 */
export const OnDevelopersHowItWorks: Story = {
  render: (args) => (
    <TemplateContainer {...developersHowItWorksContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: developersHowItWorksProps,
}

/**
 * OnProvidersHowItWorks - providersHowItWorksProps
 * @tags providers, how-it-works, earnings
 *
 * OnDevelopersHowItWorks - howItWorksProps
 * @tags developers, how-it-works, deployment
 *
 * OnHomeHowItWorks - howItWorksProps
 * @tags home, how-it-works, steps
 *
 * HowItWorks as used on the Providers page
 * - Four-step guide to start earning
 * - Terminal installation
 * - Configuration and marketplace setup
 * - Payout tracking
 */
export const OnProvidersHowItWorks: Story = {
  render: (args) => (
    <TemplateContainer {...providersHowItWorksContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: providersHowItWorksProps,
}
