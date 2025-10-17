import { comparisonTemplateContainerProps, comparisonTemplateProps } from '../../pages/HomePage/HomePageProps'
import {
  enterpriseComparisonContainerProps,
  enterpriseComparisonProps,
} from '../../pages/EnterprisePage/EnterprisePageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { ComparisonTemplate } from './ComparisonTemplate'

const meta = {
  title: 'Templates/ComparisonTemplate',
  component: ComparisonTemplate,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ComparisonTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnEnterpriseComparison - enterpriseComparisonProps
 * @tags enterprise, comparison, compliance
 * 
 * OnHomeComparison - comparisonTemplateProps
 * @tags home, comparison, features
 * 
 * ComparisonTemplate as used on the Home page
 */
export const OnHomeComparison: Story = {
  render: (args) => (
    <TemplateContainer {...comparisonTemplateContainerProps}>
      <ComparisonTemplate {...args} />
    </TemplateContainer>
  ),
  args: comparisonTemplateProps,
}

/**
 * OnEnterpriseComparison - enterpriseComparisonProps
 * @tags enterprise, comparison, compliance
 * 
 * OnHomeComparison - comparisonTemplateProps
 * @tags home, comparison, features
 * 
 * ComparisonTemplate as used on the Enterprise page
 */
export const OnEnterpriseComparison: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseComparisonContainerProps}>
      <ComparisonTemplate {...args} />
    </TemplateContainer>
  ),
  args: enterpriseComparisonProps,
}
