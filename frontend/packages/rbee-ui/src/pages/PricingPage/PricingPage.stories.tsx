import type { Meta, StoryObj } from '@storybook/react'
import PricingPage from './PricingPage'

const meta = {
  title: 'Pages/PricingPage',
  component: PricingPage,
  parameters: { layout: 'fullscreen' },
} satisfies Meta<typeof PricingPage>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}
