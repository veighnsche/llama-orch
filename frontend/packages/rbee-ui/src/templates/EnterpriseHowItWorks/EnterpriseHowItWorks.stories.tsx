import { enterpriseHowItWorksContainerProps, enterpriseHowItWorksProps } from '@rbee/ui/pages/EnterprisePage'
import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { EnterpriseHowItWorks } from './EnterpriseHowItWorks'

const meta = {
  title: 'Templates/EnterpriseHowItWorks',
  component: EnterpriseHowItWorks,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EnterpriseHowItWorks>

export default meta
type Story = StoryObj<typeof meta>

/**
 * EnterpriseHowItWorks as used on the Enterprise page
 * - Deployment process timeline
 * - Step-by-step implementation guide
 * - Enterprise onboarding flow
 */
export const OnEnterprisePage: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseHowItWorksContainerProps}>
      <EnterpriseHowItWorks {...args} />
    </TemplateContainer>
  ),
  args: enterpriseHowItWorksProps,
}
