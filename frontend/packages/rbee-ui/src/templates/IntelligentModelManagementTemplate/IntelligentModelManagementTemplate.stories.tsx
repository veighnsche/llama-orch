import type { Meta, StoryObj } from '@storybook/react'
import { IntelligentModelManagementTemplate } from './IntelligentModelManagementTemplate'
import { intelligentModelManagementProps } from '@rbee/ui/pages/FeaturesPage'

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
  args: intelligentModelManagementProps,
}
