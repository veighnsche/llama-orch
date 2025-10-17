import { TemplateContainer } from '@rbee/ui/molecules'
import {
  solutionTemplateContainerProps as developersContainerProps,
  solutionTemplateProps as developersSolutionProps,
} from '@rbee/ui/pages/DevelopersPage'
import { enterpriseSolutionContainerProps, enterpriseSolutionProps } from '@rbee/ui/pages/EnterprisePage'
import { solutionTemplateContainerProps as homeContainerProps, solutionTemplateProps as homeSolutionTemplateProps } from '@rbee/ui/pages/HomePage'
import { educationSolutionContainerProps, educationSolutionProps } from '@rbee/ui/pages/EducationPage'
import { complianceSolutionContainerProps, complianceSolutionProps } from '@rbee/ui/pages/CompliancePage'
import { devopsSolutionContainerProps, devopsSolutionProps } from '@rbee/ui/pages/DevOpsPage'
import { solutionContainerProps, solutionProps } from '@rbee/ui/pages/ResearchPage'
import { securityDefenseLayersContainerProps, securityDefenseLayersProps } from '@rbee/ui/pages/SecurityPage'
import {
  providersMarketplaceContainerProps,
  providersMarketplaceSolutionProps,
  providersSolutionContainerProps,
  providersSolutionProps,
} from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { SolutionTemplate } from './SolutionTemplate'

const meta = {
  title: 'Templates/SolutionTemplate',
  component: SolutionTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof SolutionTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnHomeSolution - solutionTemplateProps
 * @tags home, solution, benefits, orchestration
 *
 * SolutionTemplate as used on the Home page
 * - Four benefits with topology visualization
 * - Multi-host BeeArchitecture diagram
 * - Shows CUDA, Metal, and CPU orchestration
 */
export const OnHomeSolution: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...homeContainerProps}>
      <SolutionTemplate {...homeSolutionTemplateProps} />
    </TemplateContainer>
  ),
}

/**
 * OnEnterpriseSolution - enterpriseSolutionProps
 * @tags enterprise, solution, compliance, gdpr
 *
 * SolutionTemplate as used on the Enterprise page
 * - Four compliance-focused features
 * - How It Works steps for enterprise deployment
 * - Compliance metrics card with GDPR references
 * - EU data sovereignty illustration
 * - Primary and secondary CTAs
 */
export const OnEnterpriseSolution: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...enterpriseSolutionContainerProps}>
      <SolutionTemplate {...enterpriseSolutionProps} />
    </TemplateContainer>
  ),
}

/**
 * OnDevelopersSolution - solutionTemplateProps
 * @tags developers, solution, api, openai
 *
 * SolutionTemplate as used on the Developers page
 * - Four benefits focused on developer needs
 * - How It Works steps
 * - OpenAI-compatible API code example in aside
 * - Primary and secondary CTAs
 */
export const OnDevelopersSolution: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...developersContainerProps}>
      <SolutionTemplate {...developersSolutionProps} />
    </TemplateContainer>
  ),
}

/**
 * OnProvidersSolution - providersSolutionProps
 * @tags providers, solution, gpu, earnings
 *
 * SolutionTemplate as used on the Providers page
 * - Four benefits focused on GPU providers
 * - How It Works steps for marketplace
 * - Earnings card with GPU pricing estimates
 * - Primary and secondary CTAs
 */
export const OnProvidersSolution: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...providersSolutionContainerProps}>
      <SolutionTemplate {...providersSolutionProps} />
    </TemplateContainer>
  ),
}

/**
 * OnProvidersMarketplace - providersMarketplaceSolutionProps
 * @tags providers, marketplace, commission
 *
 * SolutionTemplate for Providers Marketplace
 * - Four marketplace feature tiles
 * - How It Works steps
 * - Commission structure card as aside
 * - Used on ProvidersPage marketplace section
 */
export const OnProvidersMarketplace: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...providersMarketplaceContainerProps}>
      <SolutionTemplate {...providersMarketplaceSolutionProps} />
    </TemplateContainer>
  ),
}

/**
 * SolutionTemplate as used on the Education page
 * - Real production infrastructure for learning
 * - Hands-on with production code
 * - BDD testing and real patterns
 */
export const OnEducationSolution: Story = {
  render: (args) => (
    <TemplateContainer {...educationSolutionContainerProps}>
      <SolutionTemplate {...args} />
    </TemplateContainer>
  ),
  args: educationSolutionProps,
}

/**
 * SolutionTemplate as used on the Compliance page
 * - Compliance page usage
 */
export const OnComplianceSolution: Story = {
  render: (args) => (
    <TemplateContainer {...complianceSolutionContainerProps}>
      <SolutionTemplate {...args} />
    </TemplateContainer>
  ),
  args: complianceSolutionProps,
}

/**
 * SolutionTemplate as used on the DevOps page
 * - DevOps page usage
 */
export const OnDevOpsSolution: Story = {
  render: (args) => (
    <TemplateContainer {...devopsSolutionContainerProps}>
      <SolutionTemplate {...args} />
    </TemplateContainer>
  ),
  args: devopsSolutionProps,
}

/**
 * SolutionTemplate as used on the Research page
 * - Research page usage
 */
export const OnResearchSolution: Story = {
  render: (args) => (
    <TemplateContainer {...solutionContainerProps}>
      <SolutionTemplate {...args} />
    </TemplateContainer>
  ),
  args: solutionProps,
}

/**
 * SolutionTemplate as used on the Security page
 * - Security page usage
 */
export const OnSecuritySolution: Story = {
  render: (args) => (
    <TemplateContainer {...securityDefenseLayersContainerProps}>
      <SolutionTemplate {...args} />
    </TemplateContainer>
  ),
  args: securityDefenseLayersProps,
}
