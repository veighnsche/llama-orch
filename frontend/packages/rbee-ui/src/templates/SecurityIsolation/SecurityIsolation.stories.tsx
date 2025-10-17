import { TemplateContainer } from '@rbee/ui/molecules'
import type { Meta, StoryObj } from '@storybook/react'
import { securityIsolationContainerProps, securityIsolationProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { SecurityIsolation } from './SecurityIsolation'

const meta: Meta<typeof SecurityIsolation> = {
  title: 'Templates/SecurityIsolation',
  component: SecurityIsolation,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof SecurityIsolation>

export const OnFeaturesSecurity: Story = {
  render: (args) => (
    <TemplateContainer {...securityIsolationContainerProps}>
      <SecurityIsolation {...args} />
    </TemplateContainer>
  ),
  args: securityIsolationProps,
}
