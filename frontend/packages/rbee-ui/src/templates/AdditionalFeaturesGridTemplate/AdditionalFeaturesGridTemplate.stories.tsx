import type { Meta, StoryObj } from '@storybook/react'
import { AdditionalFeaturesGridTemplate } from './AdditionalFeaturesGridTemplate'
import { additionalFeaturesGridProps } from '../../pages/FeaturesPage/FeaturesPageProps'

const meta: Meta<typeof AdditionalFeaturesGridTemplate> = {
  title: 'Templates/AdditionalFeaturesGridTemplate',
  component: AdditionalFeaturesGridTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof AdditionalFeaturesGridTemplate>

export const OnFeaturesPage: Story = {
  args: additionalFeaturesGridProps,
}
