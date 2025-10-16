import { TemplateContainer } from '@rbee/ui/molecules'
import { providersTestimonialsContainerProps, providersTestimonialsProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersTestimonialsTemplate } from './ProvidersTestimonialsTemplate'

const meta = {
  title: 'Templates/ProvidersTestimonialsTemplate',
  component: ProvidersTestimonialsTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersTestimonialsTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersTestimonialsTemplate as used on the Providers page
 * - Features testimonials from existing GPU providers
 * - Showcases success stories and earnings
 * - Builds trust through social proof
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersTestimonialsContainerProps}>
      <ProvidersTestimonialsTemplate {...args} />
    </TemplateContainer>
  ),
  args: providersTestimonialsProps,
}
