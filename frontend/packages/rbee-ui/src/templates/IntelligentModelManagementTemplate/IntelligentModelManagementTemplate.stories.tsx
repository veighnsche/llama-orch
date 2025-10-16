import type { Meta, StoryObj } from '@storybook/react'
import { IntelligentModelManagementTemplate } from './IntelligentModelManagementTemplate'
import { intelligentModelManagementProps } from '../../pages/FeaturesPage/FeaturesPageProps'

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
