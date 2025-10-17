import { TemplateContainer } from '@rbee/ui/molecules'
import { enterpriseComplianceContainerProps, enterpriseComplianceProps } from '@rbee/ui/pages/EnterprisePage'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseCompliance } from './EnterpriseCompliance'

const meta = {
  title: 'Templates/EnterpriseCompliance',
  component: EnterpriseCompliance,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseCompliance>

export default meta
type Story = StoryObj<typeof meta>

/**
 * OnEnterpriseCompliance - enterpriseComplianceProps
 * @tags enterprise, compliance, gdpr
 *
 * EnterpriseCompliance as used on the Enterprise page
 * - GDPR, SOC2, ISO 27001 compliance pillars
 * - Detailed compliance features
 * - Audit retention and data residency
 */
export const OnEnterpriseCompliance: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseComplianceContainerProps}>
      <EnterpriseCompliance {...args} />
    </TemplateContainer>
  ),
  args: enterpriseComplianceProps,
}
