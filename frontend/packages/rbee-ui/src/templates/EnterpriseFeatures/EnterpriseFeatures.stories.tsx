import { enterpriseFeaturesContainerProps, enterpriseFeaturesProps } from '@rbee/ui/pages/EnterprisePage'
import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseFeatures } from './EnterpriseFeatures'

const meta = {
  title: 'Templates/EnterpriseFeatures',
  component: EnterpriseFeatures,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseFeatures>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EnterpriseFeatures as used on the Enterprise page
 * - Enterprise capabilities grid
 * - Advanced features for regulated industries
 * - Security and compliance focus
 */
export const OnEnterprisePage: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseFeaturesContainerProps}>
      <EnterpriseFeatures {...args} />
    </TemplateContainer>
  ),
  args: enterpriseFeaturesProps,
}
