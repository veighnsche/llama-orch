import type { Meta, StoryObj } from '@storybook/react'
import { MonthlyEarningsPanel } from './MonthlyEarningsPanel'

const meta = {
  title: 'Organisms/MonthlyEarningsPanel',
  component: MonthlyEarningsPanel,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    progressPercentage: {
      control: { type: 'range', min: 0, max: 100, step: 1 },
    },
  },
} satisfies Meta<typeof MonthlyEarningsPanel>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    monthLabel: 'This Month',
    monthEarnings: '€2,847',
    monthGrowth: '+23% vs last month',
    progressPercentage: 68,
  },
}

export const HighEarnings: Story = {
  args: {
    monthLabel: 'This Month',
    monthEarnings: '€8,432',
    monthGrowth: '+156% vs last month',
    progressPercentage: 92,
  },
}

export const LowProgress: Story = {
  args: {
    monthLabel: 'This Month',
    monthEarnings: '€1,234',
    monthGrowth: '+12% vs last month',
    progressPercentage: 25,
  },
}

export const NegativeGrowth: Story = {
  args: {
    monthLabel: 'This Month',
    monthEarnings: '€1,847',
    monthGrowth: '-8% vs last month',
    progressPercentage: 45,
  },
}

export const CustomMonth: Story = {
  args: {
    monthLabel: 'January 2025',
    monthEarnings: '$5,621',
    monthGrowth: '+34% vs December',
    progressPercentage: 78,
  },
}
