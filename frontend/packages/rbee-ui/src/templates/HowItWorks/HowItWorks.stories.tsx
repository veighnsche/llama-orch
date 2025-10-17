import { TemplateContainer } from '@rbee/ui/molecules'
import {
  howItWorksContainerProps as developersHowItWorksContainerProps,
  howItWorksProps as developersHowItWorksProps,
} from '@rbee/ui/pages/DevelopersPage'
import { howItWorksContainerProps as homeHowItWorksContainerProps, howItWorksProps as homeHowItWorksProps } from '@rbee/ui/pages/HomePage'
import { providersHowItWorksContainerProps, providersHowItWorksProps } from '@rbee/ui/pages/ProvidersPage'
import { educationLabExercisesContainerProps, educationLabExercisesProps } from '@rbee/ui/pages/EducationPage'
import { howToContributeContainerProps, howToContributeProps } from '@rbee/ui/pages/CommunityPage'
import { workflowContainerProps, workflowProps } from '@rbee/ui/pages/ResearchPage'
import { securityVulnerabilityDisclosureContainerProps, securityVulnerabilityDisclosureProps } from '@rbee/ui/pages/SecurityPage'
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
    <TemplateContainer {...homeHowItWorksContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: homeHowItWorksProps,
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

/**
 * HowItWorks as used on the Education page
 * - Step-by-step hands-on labs
 * - Deploy workers, orchestrate, monitor
 * - Write BDD tests
 */
export const OnEducationLabExercises: Story = {
  render: (args) => (
    <TemplateContainer {...educationLabExercisesContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: educationLabExercisesProps,
}

/**
 * HowItWorks as used on the Community page
 * - Community page usage
 */
export const OnCommunityHowItWorks: Story = {
  render: (args) => (
    <TemplateContainer {...howToContributeContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: howToContributeProps,
}

/**
 * HowItWorks as used on the Research page
 * - Research page usage
 */
export const OnResearchHowItWorks: Story = {
  render: (args) => (
    <TemplateContainer {...workflowContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: workflowProps,
}

/**
 * HowItWorks as used on the Security page
 * - Security page usage
 */
export const OnSecurityHowItWorks: Story = {
  render: (args) => (
    <TemplateContainer {...securityVulnerabilityDisclosureContainerProps}>
      <HowItWorks {...args} />
    </TemplateContainer>
  ),
  args: securityVulnerabilityDisclosureProps,
}
