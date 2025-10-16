import type { Meta, StoryObj } from '@storybook/react'
import { CrossNodeOrchestrationTemplate } from './CrossNodeOrchestrationTemplate'
import { crossNodeOrchestrationProps } from '@rbee/ui/pages/FeaturesPage'

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
