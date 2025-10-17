import type { Meta, StoryObj } from '@storybook/react'
import { crossNodeOrchestrationProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { CrossNodeOrchestration } from './CrossNodeOrchestration'

const meta: Meta<typeof CrossNodeOrchestration> = {
  title: 'Templates/CrossNodeOrchestration',
  component: CrossNodeOrchestration,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof CrossNodeOrchestration>

export const OnFeaturesPage: Story = {
  args: crossNodeOrchestrationProps,
}
