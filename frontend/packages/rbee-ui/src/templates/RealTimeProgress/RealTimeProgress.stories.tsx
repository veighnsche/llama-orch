import type { Meta, StoryObj } from '@storybook/react'
import { realTimeProgressContainerProps, realTimeProgressProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import { RealTimeProgress } from './RealTimeProgress'

const meta: Meta<typeof RealTimeProgress> = {
  title: 'Templates/RealTimeProgress',
  component: RealTimeProgress,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof RealTimeProgress>

export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...realTimeProgressContainerProps}>
      <RealTimeProgress {...args} />
    </TemplateContainer>
  ),
  args: realTimeProgressProps,
}
