import type { Meta, StoryObj } from '@storybook/react'
import { providersEarningsContainerProps, providersEarningsProps } from '@rbee/ui/pages/ProvidersPage'
import { TemplateContainer } from '@rbee/ui/molecules'
import { ProvidersEarningsTemplate } from './ProvidersEarningsTemplate'

const meta = {
  title: 'Templates/ProvidersEarningsTemplate',
  component: ProvidersEarningsTemplate,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersEarningsTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersEarningsTemplate as used on the Providers page
 * - Displays earnings calculator and potential revenue
 * - Shows pricing tiers and payout structures
 * - Features transparent commission breakdown
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersEarningsContainerProps}>
      <ProvidersEarningsTemplate {...args} />
    </TemplateContainer>
  ),
  args: providersEarningsProps,
}
