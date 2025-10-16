import type { Meta, StoryObj } from '@storybook/react'
import { RealTimeProgressTemplate } from './RealTimeProgressTemplate'
import { realTimeProgressProps } from '../../pages/FeaturesPage/FeaturesPageProps'

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
  args: realTimeProgressProps,
}
