import type { Meta, StoryObj } from '@storybook/react'
import { crossNodeOrchestrationContainerProps, crossNodeOrchestrationProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
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

export const OnFeaturesCrossNode: Story = {
  render: (args) => (
    <TemplateContainer {...crossNodeOrchestrationContainerProps}>
      <CrossNodeOrchestration {...args} />
    </TemplateContainer>
  ),
  args: crossNodeOrchestrationProps,
}
