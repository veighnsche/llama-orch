import type { Meta, StoryObj } from '@storybook/react'
import { MultiBackendGpuTemplate } from './MultiBackendGpuTemplate'
import { multiBackendGpuProps } from '../../pages/FeaturesPage/FeaturesPageProps'

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
