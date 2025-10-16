import type { Meta, StoryObj } from '@storybook/react'
import { providersMarketplaceContainerProps, providersMarketplaceProps } from '@rbee/ui/pages/ProvidersPage'
import { TemplateContainer } from '@rbee/ui/molecules'
import { ProvidersMarketplaceTemplate } from './ProvidersMarketplaceTemplate'

const meta = {
  title: 'Templates/ProvidersMarketplaceTemplate',
  component: ProvidersMarketplaceTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersMarketplaceTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersMarketplaceTemplate as used on the Providers page
 * - Explains the marketplace model and listing process
 * - Shows how providers connect with customers
 * - Features marketplace benefits and competitive advantages
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersMarketplaceContainerProps}>
      <ProvidersMarketplaceTemplate {...args} />
    </TemplateContainer>
  ),
  args: providersMarketplaceProps,
}
