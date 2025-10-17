import { TemplateContainer } from '@rbee/ui/molecules'
import { enterpriseHowItWorksContainerProps, enterpriseHowItWorksProps } from '@rbee/ui/pages/EnterprisePage'
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
 * OnEnterpriseHowItWorks - enterpriseHowItWorksProps
 * @tags enterprise, deployment, process
 *
 * EnterpriseHowItWorks as used on the Enterprise page
 * - Deployment process timeline
 * - Step-by-step implementation guide
 * - Enterprise onboarding flow
 */
export const OnEnterpriseHowItWorks: Story = {
  render: (args) => (
    <TemplateContainer {...enterpriseHowItWorksContainerProps}>
      <EnterpriseHowItWorks {...args} />
    </TemplateContainer>
  ),
  args: enterpriseHowItWorksProps,
}
