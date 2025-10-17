import type { Meta, StoryObj } from '@storybook/react'
import { intelligentModelManagementContainerProps, intelligentModelManagementProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import { IntelligentModelManagement } from './IntelligentModelManagement'

const meta: Meta<typeof IntelligentModelManagement> = {
  title: 'Templates/IntelligentModelManagement',
  component: IntelligentModelManagement,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof IntelligentModelManagement>

export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...intelligentModelManagementContainerProps}>
      <IntelligentModelManagement {...args} />
    </TemplateContainer>
  ),
  args: intelligentModelManagementProps,
}
