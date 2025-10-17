import { enterpriseUseCasesContainerProps, enterpriseUseCasesProps } from '@rbee/ui/pages/EnterprisePage'
import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseUseCases } from './EnterpriseUseCases'

const meta = {
  title: 'Templates/EnterpriseUseCases',
  component: EnterpriseUseCases,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseUseCases>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EnterpriseUseCases as used on the Enterprise page
 * - Industry playbooks grid
 * - Finance, Healthcare, Legal, Government use cases
 * - Sector-specific compliance requirements
 */
export const OnEnterprisePage: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseUseCasesContainerProps}>
      <EnterpriseUseCases {...args} />
    </TemplateContainer>
  ),
  args: enterpriseUseCasesProps,
}
