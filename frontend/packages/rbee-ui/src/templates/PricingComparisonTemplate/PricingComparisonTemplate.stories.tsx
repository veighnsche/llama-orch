import { pricingComparisonProps } from '@rbee/ui/pages/PricingPage/PricingPageProps'
import type { Meta, StoryObj } from '@storybook/react'
import { PricingComparisonTemplate } from './PricingComparisonTemplate'

const meta = {
  title: 'Templates/PricingComparisonTemplate',
  component: PricingComparisonTemplate,
  parameters: { layout: 'padded' },
  tags: ['autodocs'],
} satisfies Meta<typeof PricingComparisonTemplate>

export default meta
type Story = StoryObj<typeof meta>

/**
 * PricingComparisonTemplate as used on the Pricing page
 * - Detailed feature comparison table
 * - Three columns: Home/Lab, Team, Enterprise
 * - Three groups: Core Platform, Productivity, Support & Services
 * - CTA buttons at bottom
 */
export const OnPricingPage: Story = {
  args: pricingComparisonProps,
}
