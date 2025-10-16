import { howItWorksProps } from '@rbee/ui/pages/HomePage'
import type { Meta, StoryObj } from '@storybook/react'
import { HowItWorks } from './HowItWorks'

const meta = {
  title: 'Templates/HowItWorks',
  component: HowItWorks,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof HowItWorks>

export default meta
type Story = StoryObj<typeof meta>

// Use props from HomePage - single source of truth
export const OnHomePage: Story = {
  args: howItWorksProps,
}
