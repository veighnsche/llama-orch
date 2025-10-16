import type { Meta, StoryObj } from '@storybook/react'
import { crossNodeOrchestrationProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { CrossNodeOrchestrationTemplate } from './CrossNodeOrchestrationTemplate'

const meta: Meta<typeof CrossNodeOrchestrationTemplate> = {
  title: 'Templates/CrossNodeOrchestrationTemplate',
  component: CrossNodeOrchestrationTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof CrossNodeOrchestrationTemplate>

export const OnFeaturesPage: Story = {
  args: crossNodeOrchestrationProps,
}
