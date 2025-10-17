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
 * ComparisonTemplate as used on the Home page
 */
export const OnHomePage: Story = {
  render: (args) => (
    <TemplateContainer {...comparisonTemplateContainerProps}>
      <ComparisonTemplate {...args} />
    </TemplateContainer>
  ),
  args: comparisonTemplateProps,
}

/**
 * ComparisonTemplate as used on the Enterprise page
 */
export const OnEnterprisePage: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseComparisonContainerProps}>
      <ComparisonTemplate {...args} />
    </TemplateContainer>
  ),
  args: enterpriseComparisonProps,
}
