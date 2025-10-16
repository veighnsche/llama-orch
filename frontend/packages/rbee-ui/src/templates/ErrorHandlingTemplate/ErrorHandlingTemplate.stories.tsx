import type { Meta, StoryObj } from '@storybook/react'
import { errorHandlingProps } from '../../pages/FeaturesPage/FeaturesPageProps'
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
  args: errorHandlingProps,
}
