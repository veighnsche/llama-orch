import { TemplateContainer } from '@rbee/ui/molecules'
import { providersEarningsContainerProps, providersEarningsProps } from '@rbee/ui/pages/ProvidersPage'
import type { Meta, StoryObj } from '@storybook/react'
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
 * OnProvidersEarnings - providersEarningsProps
 * @tags providers, earnings, calculator
 * 
 * ProvidersEarnings as used on the Providers page
 * - Displays earnings calculator and potential revenue
 * - Shows pricing tiers and payout structures
 * - Features transparent commission breakdown
 */
export const OnProvidersEarnings: Story = {
  render: (args) => (
    <TemplateContainer {...providersEarningsContainerProps}>
      <ProvidersEarnings {...args} />
    </TemplateContainer>
  ),
  args: providersEarningsProps,
}
