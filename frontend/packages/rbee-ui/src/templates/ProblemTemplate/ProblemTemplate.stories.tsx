import { TemplateContainer } from '@rbee/ui/molecules'
import {
  problemTemplateContainerProps as developersContainerProps,
  problemTemplateProps as developersProblems,
} from '@rbee/ui/pages/DevelopersPage'
import { enterpriseProblemTemplateContainerProps, enterpriseProblemTemplateProps } from '@rbee/ui/pages/EnterprisePage'
import { problemTemplateProps } from '@rbee/ui/pages/HomePage'
import { providersProblemContainerProps, providersProblemProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { ProblemTemplate } from './ProblemTemplate'

const meta = {
  title: 'Templates/ProblemTemplate',
  component: ProblemTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProblemTemplate>

export default meta
type Story = StoryObj<typeof ProblemTemplate>

/**
 * OnHomeProblem - problemTemplateProps
 * @tags home, problem, risks, ai-dependency
 *
 * ProblemTemplate as used on the Home page
 * - AI provider dependency risks
 * - Model changes, pricing changes, shutdown risks
 * - Codebase maintainability concerns
 */
export const OnHomeProblem: Story = {
  render: () => (
    <TemplateContainer
      title="The Hidden Risk of AI Dependency"
      description="Building with AI assistance creates dependencies you can't control."
      background={{ variant: 'background' }}
      paddingY="2xl"
      maxWidth="6xl"
      align="center"
    >
      <ProblemTemplate {...problemTemplateProps} />
    </TemplateContainer>
  ),
}

/**
 * OnDevelopersProblem - problemTemplateProps
 * @tags developers, problem, vendor-lock-in
 *
 * ProblemTemplate as used on the Developers page
 * - Hidden risks of AI-assisted development
 * - Model changes, pricing unpredictability
 * - Vendor lock-in and shutdown risks
 */
export const OnDevelopersProblem: Story = {
  render: () => (
    <TemplateContainer {...developersContainerProps}>
      <ProblemTemplate {...developersProblems} />
    </TemplateContainer>
  ),
}

/**
 * OnEnterpriseProblem - enterpriseProblemTemplateProps
 * @tags enterprise, problem, compliance, gdpr
 *
 * ProblemTemplate as used on the Enterprise page
 * - Compliance challenges of cloud AI
 * - Data sovereignty, audit trail, GDPR risks
 * - Regulatory compliance concerns
 */
export const OnEnterpriseProblem: Story = {
  render: () => (
    <TemplateContainer {...enterpriseProblemTemplateContainerProps}>
      <ProblemTemplate {...enterpriseProblemTemplateProps} />
    </TemplateContainer>
  ),
}

/**
 * OnProvidersProblem - providersProblemProps
 * @tags providers, problem, gpu, idle-hardware
 *
 * ProblemTemplate as used on the Providers page
 * - GPU provider challenges
 * - Idle hardware, revenue loss, complexity
 * - Monetization and trust concerns
 */
export const OnProvidersProblem: Story = {
  render: () => (
    <TemplateContainer {...providersProblemContainerProps}>
      <ProblemTemplate {...providersProblemProps} />
    </TemplateContainer>
  ),
}
