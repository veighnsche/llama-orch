import type { Meta, StoryObj } from '@storybook/react'
import { multiBackendGpuProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { MultiBackendGpuTemplate } from './MultiBackendGpuTemplate'

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
