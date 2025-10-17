import type { Meta, StoryObj } from '@storybook/react'
import { RibbonBanner } from './RibbonBanner'

const meta: Meta<typeof RibbonBanner> = {
  title: 'Organisms/Footers/RibbonBanner',
  component: RibbonBanner,
  parameters: {
    layout: 'centered',
  },
}

export default meta
type Story = StoryObj<typeof RibbonBanner>

export const Default: Story = {
  args: {
    text: '€2M Insurance Coverage • All GPU Providers Verified',
  },
}

export const Short: Story = {
  args: {
    text: 'Insured up to €2M',
  },
}

export const Long: Story = {
  args: {
    text: 'All transactions protected with €2M insurance coverage through our verified provider network',
  },
}
