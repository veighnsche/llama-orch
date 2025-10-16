import type { Meta, StoryObj } from '@storybook/react'
import { additionalFeaturesGridProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { AdditionalFeaturesGridTemplate } from './AdditionalFeaturesGridTemplate'

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
