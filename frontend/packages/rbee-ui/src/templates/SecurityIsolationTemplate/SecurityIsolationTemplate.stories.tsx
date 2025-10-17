import type { Meta, StoryObj } from '@storybook/react'
import { securityIsolationContainerProps, securityIsolationProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import { SecurityIsolationTemplate } from './SecurityIsolationTemplate'

const meta: Meta<typeof SecurityIsolationTemplate> = {
  title: 'Templates/SecurityIsolationTemplate',
  component: SecurityIsolationTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof SecurityIsolationTemplate>

export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...securityIsolationContainerProps}>
      <SecurityIsolationTemplate {...args} />
    </TemplateContainer>
  ),
  args: securityIsolationProps,
}
