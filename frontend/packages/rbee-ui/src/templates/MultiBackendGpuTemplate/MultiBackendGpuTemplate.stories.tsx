import type { Meta, StoryObj } from '@storybook/react'
import { MultiBackendGpuTemplate } from './MultiBackendGpuTemplate'
import { multiBackendGpuProps } from '@rbee/ui/pages/FeaturesPage'

const meta: Meta<typeof MultiBackendGpuTemplate> = {
  title: 'Templates/MultiBackendGpuTemplate',
  component: MultiBackendGpuTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof MultiBackendGpuTemplate>

export const OnFeaturesPage: Story = {
  args: multiBackendGpuProps,
}
