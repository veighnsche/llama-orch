import type { Meta, StoryObj } from '@storybook/react'
import { realTimeProgressContainerProps, realTimeProgressProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import { RealTimeProgressTemplate } from './RealTimeProgressTemplate'

const meta: Meta<typeof RealTimeProgressTemplate> = {
  title: 'Templates/RealTimeProgressTemplate',
  component: RealTimeProgressTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof RealTimeProgressTemplate>

export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...realTimeProgressContainerProps}>
      <RealTimeProgressTemplate {...args} />
    </TemplateContainer>
  ),
  args: realTimeProgressProps,
}
