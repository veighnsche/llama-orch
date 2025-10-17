import { TemplateContainer } from '@rbee/ui/molecules'
import { enterpriseFeaturesContainerProps, enterpriseFeaturesGridProps } from '@rbee/ui/pages/EnterprisePage'
import { educationResourcesGridContainerProps, educationResourcesGridProps } from '@rbee/ui/pages/EducationPage'
import {
  providersSecurityContainerProps,
  providersSecurityGridProps,
  providersUseCasesContainerProps,
  providersUseCasesGridProps,
} from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { CardGridTemplate } from './CardGridTemplate'

const meta = {
  title: 'Templates/CardGridTemplate',
  component: CardGridTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof CardGridTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnProvidersUseCases - providersUseCasesProps
 * @tags providers, use-cases, personas, grid
 *
 * Provider use case cards in 2-column grid
 */
export const OnProvidersUseCases: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...providersUseCasesContainerProps}>
      <CardGridTemplate {...providersUseCasesGridProps} />
    </TemplateContainer>
  ),
}

/**
 * OnProvidersSecurity - providersSecurityProps
 * @tags providers, security, trust, grid
 *
 * Provider security cards in 2-column grid
 */
export const OnProvidersSecurity: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...providersSecurityContainerProps}>
      <CardGridTemplate {...providersSecurityGridProps} />
    </TemplateContainer>
  ),
}

/**
 * OnEnterpriseFeatures - enterpriseFeaturesData
 * @tags enterprise, features, capabilities, grid
 *
 * Enterprise feature cards in 2-column grid
 */
export const OnEnterpriseFeatures: Story = {
  args: {} as any,
  render: () => (
    <TemplateContainer {...enterpriseFeaturesContainerProps}>
      <CardGridTemplate {...enterpriseFeaturesGridProps} />
    </TemplateContainer>
  ),
}

/**
 * CardGridTemplate as used on the Education page
 * - Learning resources
 * - Documentation, examples, tutorials
 * - Community support
 */
export const OnEducationResourcesGrid: Story = {
  render: (args) => (
    <TemplateContainer {...educationResourcesGridContainerProps}>
      <CardGridTemplate {...args} />
    </TemplateContainer>
  ),
  args: educationResourcesGridProps,
}
