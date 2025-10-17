import { TemplateContainer } from '@rbee/ui/molecules'
import {
  solutionTemplateContainerProps as developersContainerProps,
  solutionTemplateProps as developersSolutionProps,
} from '@rbee/ui/pages/DevelopersPage'
import {
  enterpriseSolutionContainerProps,
  enterpriseSolutionProps,
} from '@rbee/ui/pages/EnterprisePage'
import {
  solutionTemplateContainerProps as homeContainerProps,
  solutionTemplateProps,
} from '@rbee/ui/pages/HomePage'
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
 * SolutionTemplate as used on the Home page
 * - Four benefits with topology visualization
 * - Multi-host BeeArchitecture diagram
 * - Shows CUDA, Metal, and CPU orchestration
 */
export const OnHomePage: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...homeContainerProps}>
      <SolutionTemplate {...solutionTemplateProps} />
    </TemplateContainer>
  ),
}

/**
 * SolutionTemplate as used on the Enterprise page
 * - Four compliance-focused features
 * - How It Works steps for enterprise deployment
 * - Compliance metrics card with GDPR references
 * - EU data sovereignty illustration
 * - Primary and secondary CTAs
 */
export const OnEnterprisePage: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...enterpriseSolutionContainerProps}>
      <SolutionTemplate {...enterpriseSolutionProps} />
    </TemplateContainer>
  ),
}

/**
 * SolutionTemplate as used on the Developers page
 * - Four benefits focused on developer needs
 * - How It Works steps
 * - OpenAI-compatible API code example in aside
 * - Primary and secondary CTAs
 */
export const OnDevelopersPage: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...developersContainerProps}>
      <SolutionTemplate {...developersSolutionProps} />
    </TemplateContainer>
  ),
}

/**
 * SolutionTemplate as used on the Providers page
 * - Four benefits focused on GPU providers
 * - How It Works steps for marketplace
 * - Earnings card with GPU pricing estimates
 * - Primary and secondary CTAs
 */
export const OnProvidersPage: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...providersSolutionContainerProps}>
      <SolutionTemplate {...providersSolutionProps} />
    </TemplateContainer>
  ),
}

/**
 * SolutionTemplate for Providers Marketplace
 * - Four marketplace feature tiles
 * - How It Works steps
 * - Commission structure card as aside
 * - Used on ProvidersPage marketplace section
 */
export const ProvidersMarketplace: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...providersMarketplaceContainerProps}>
      <SolutionTemplate {...providersMarketplaceSolutionProps} />
    </TemplateContainer>
  ),
}
