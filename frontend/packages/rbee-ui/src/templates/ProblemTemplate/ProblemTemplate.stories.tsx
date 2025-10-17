import { TemplateContainer } from '@rbee/ui/molecules'
import { complianceProblemTemplateContainerProps, complianceProblemTemplateProps } from '@rbee/ui/pages/CompliancePage'
import {
  problemTemplateContainerProps as developersContainerProps,
  problemTemplateProps as developersProblems,
} from '@rbee/ui/pages/DevelopersPage'
import { devopsProblemContainerProps, devopsProblemProps } from '@rbee/ui/pages/DevOpsPage'
import { educationProblemTemplateContainerProps, educationProblemTemplateProps } from '@rbee/ui/pages/EducationPage'
import { enterpriseProblemTemplateContainerProps, enterpriseProblemTemplateProps } from '@rbee/ui/pages/EnterprisePage'
import { problemTemplateProps as homeProblemTemplateProps } from '@rbee/ui/pages/HomePage'
import { providersProblemContainerProps, providersProblemProps } from '@rbee/ui/pages/ProvidersPage'
import { problemContainerProps, problemProps } from '@rbee/ui/pages/ResearchPage'
import { securityThreatModelContainerProps, securityThreatModelProps } from '@rbee/ui/pages/SecurityPage'
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
      <ProblemTemplate {...homeProblemTemplateProps} />
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

/**
 * OnEducationProblem - educationProblemTemplateProps
 * @tags education, problem, learning-gap, theoretical
 *
 * ProblemTemplate as used on the Education page
 * - The learning gap in distributed systems
 * - Theoretical-only education challenges
 * - No real infrastructure access
 */
export const OnEducationProblem: Story = {
  render: () => (
    <TemplateContainer {...educationProblemTemplateContainerProps}>
      <ProblemTemplate {...educationProblemTemplateProps} />
    </TemplateContainer>
  ),
}

/**
 * ProblemTemplate as used on the Compliance page
 * - Compliance page usage
 */
export const OnComplianceProblem: Story = {
  render: (args) => (
    <TemplateContainer {...complianceProblemTemplateContainerProps}>
      <ProblemTemplate {...args} />
    </TemplateContainer>
  ),
  args: complianceProblemTemplateProps,
}

/**
 * ProblemTemplate as used on the DevOps page
 * - DevOps page usage
 */
export const OnDevOpsProblem: Story = {
  render: (args) => (
    <TemplateContainer {...devopsProblemContainerProps}>
      <ProblemTemplate {...args} />
    </TemplateContainer>
  ),
  args: devopsProblemProps,
}

/**
 * ProblemTemplate as used on the Research page
 * - Research page usage
 */
export const OnResearchProblem: Story = {
  render: (args) => (
    <TemplateContainer {...problemContainerProps}>
      <ProblemTemplate {...args} />
    </TemplateContainer>
  ),
  args: problemProps,
}

/**
 * ProblemTemplate as used on the Security page
 * - Security page usage
 */
export const OnSecurityProblem: Story = {
  render: (args) => (
    <TemplateContainer {...securityThreatModelContainerProps}>
      <ProblemTemplate {...args} />
    </TemplateContainer>
  ),
  args: securityThreatModelProps,
}
