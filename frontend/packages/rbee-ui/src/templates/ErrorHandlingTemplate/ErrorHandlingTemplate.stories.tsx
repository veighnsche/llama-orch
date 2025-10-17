import type { Meta, StoryObj } from '@storybook/react'
import { errorHandlingContainerProps, errorHandlingProps } from '../../pages/FeaturesPage/FeaturesPageProps'
import { TemplateContainer } from '@rbee/ui/molecules'
import { ErrorHandlingTemplate } from './ErrorHandlingTemplate'

const meta: Meta<typeof ErrorHandlingTemplate> = {
  title: 'Templates/ErrorHandlingTemplate',
  component: ErrorHandlingTemplate,
  parameters: {
    layout: 'fullscreen',
  },
}

export default meta
type Story = StoryObj<typeof ErrorHandlingTemplate>

export const OnFeaturesPage: Story = {
  render: (args) => (
    <TemplateContainer {...errorHandlingContainerProps}>
      <ErrorHandlingTemplate {...args} />
    </TemplateContainer>
  ),
  args: errorHandlingProps,
}
