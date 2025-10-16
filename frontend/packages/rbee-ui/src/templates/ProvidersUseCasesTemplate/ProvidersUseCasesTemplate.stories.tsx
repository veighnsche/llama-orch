import { TemplateContainer } from '@rbee/ui/molecules'
import { providersUseCasesContainerProps, providersUseCasesProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersUseCasesTemplate } from './ProvidersUseCasesTemplate'

const meta = {
  title: 'Templates/ProvidersUseCasesTemplate',
  component: ProvidersUseCasesTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersUseCasesTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersUseCasesTemplate as used on the Providers page
 * - Showcases different provider types and use cases
 * - Highlights earning opportunities for various GPU setups
 * - Features provider testimonials and success stories
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersUseCasesContainerProps}>
      <ProvidersUseCasesTemplate {...args} />
    </TemplateContainer>
  ),
  args: providersUseCasesProps,
}
