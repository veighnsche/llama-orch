import type { Meta, StoryObj } from '@storybook/react'
import { intelligentModelManagementProps } from '../../pages/FeaturesPage/FeaturesPageProps'
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
  args: intelligentModelManagementProps,
}
