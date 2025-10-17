import type { Meta, StoryObj } from '@storybook/react'
import { additionalFeaturesGridContainerProps, additionalFeaturesGridProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import { AdditionalFeaturesGrid } from './AdditionalFeaturesGrid'

const meta: Meta<typeof AdditionalFeaturesGrid> = {
  title: 'Templates/AdditionalFeaturesGrid',
  component: AdditionalFeaturesGrid,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof AdditionalFeaturesGrid>

export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...additionalFeaturesGridContainerProps}>
      <AdditionalFeaturesGrid {...args} />
    </TemplateContainer>
  ),
  args: additionalFeaturesGridProps,
}
