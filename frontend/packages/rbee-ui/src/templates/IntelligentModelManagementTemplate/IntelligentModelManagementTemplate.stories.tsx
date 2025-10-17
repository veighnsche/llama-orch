import type { Meta, StoryObj } from '@storybook/react'
import { intelligentModelManagementContainerProps, intelligentModelManagementProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import { IntelligentModelManagementTemplate } from './IntelligentModelManagementTemplate'

const meta: Meta<typeof IntelligentModelManagementTemplate> = {
  title: 'Templates/IntelligentModelManagementTemplate',
  component: IntelligentModelManagementTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof IntelligentModelManagementTemplate>

export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...intelligentModelManagementContainerProps}>
      <IntelligentModelManagementTemplate {...args} />
    </TemplateContainer>
  ),
  args: intelligentModelManagementProps,
}
