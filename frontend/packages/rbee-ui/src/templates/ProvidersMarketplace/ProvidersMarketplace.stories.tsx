import type { Meta, StoryObj } from '@storybook/react'
import { providersMarketplaceContainerProps, providersMarketplaceProps } from '@rbee/ui/pages/ProvidersPage'
import { TemplateContainer } from '@rbee/ui/molecules'
import { ProvidersMarketplace } from './ProvidersMarketplace'

const meta = {
  title: 'Templates/ProvidersMarketplace',
  component: ProvidersMarketplace,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersMarketplace>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersMarketplace as used on the Providers page
 * - Explains the marketplace model and listing process
 * - Shows how providers connect with customers
 * - Features marketplace benefits and competitive advantages
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersMarketplaceContainerProps}>
      <ProvidersMarketplace {...args} />
    </TemplateContainer>
  ),
  args: providersMarketplaceProps,
}
