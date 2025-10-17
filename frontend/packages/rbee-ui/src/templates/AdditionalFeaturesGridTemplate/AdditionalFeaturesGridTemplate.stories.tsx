import type { Meta, StoryObj } from '@storybook/react'
import { additionalFeaturesGridContainerProps, additionalFeaturesGridProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
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
  render: (args) => (
    <TemplateContainer {...additionalFeaturesGridContainerProps}>
      <AdditionalFeaturesGridTemplate {...args} />
    </TemplateContainer>
  ),
  args: additionalFeaturesGridProps,
}
