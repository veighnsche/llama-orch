import type { Meta, StoryObj } from '@storybook/react'
import { providersEarningsContainerProps, providersEarningsProps } from '@rbee/ui/pages/ProvidersPage'
import { TemplateContainer } from '@rbee/ui/molecules'
import { ProvidersEarnings } from './ProvidersEarnings'

const meta = {
  title: 'Templates/ProvidersEarnings',
  component: ProvidersEarnings,
  parameters: {
    layout: 'fullscreen',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof ProvidersEarnings>

export default meta
type Story = StoryObj<typeof meta>

/**
 * ProvidersEarnings as used on the Providers page
 * - Displays earnings calculator and potential revenue
 * - Shows pricing tiers and payout structures
 * - Features transparent commission breakdown
 */
export const OnProvidersPage: Story = {
  render: (args) => (
    <TemplateContainer {...providersEarningsContainerProps}>
      <ProvidersEarnings {...args} />
    </TemplateContainer>
  ),
  args: providersEarningsProps,
}
