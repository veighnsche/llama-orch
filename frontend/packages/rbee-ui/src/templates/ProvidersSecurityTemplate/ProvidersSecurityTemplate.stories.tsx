import { TemplateContainer } from '@rbee/ui/molecules'
import { providersSecurityContainerProps, providersSecurityProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
import { ProvidersSecurityTemplate } from './ProvidersSecurityTemplate'

const meta = {
  title: 'Templates/ProvidersSecurityTemplate',
  component: ProvidersSecurityTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersSecurityTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersSecurityTemplate as used on the Providers page
 * - Highlights security measures for GPU providers
 * - Explains isolation and protection mechanisms
 * - Features compliance and trust indicators
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersSecurityContainerProps}>
      <ProvidersSecurityTemplate {...args} />
    </TemplateContainer>
  ),
  args: providersSecurityProps,
}
