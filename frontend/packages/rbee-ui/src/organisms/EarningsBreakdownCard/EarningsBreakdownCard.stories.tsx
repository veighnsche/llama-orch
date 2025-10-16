import type { Meta, StoryObj } from '@storybook/react'
import { EarningsBreakdownCard } from './EarningsBreakdownCard'

const meta = {
  title: 'Organisms/EarningsBreakdownCard',
  component: EarningsBreakdownCard,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof EarningsBreakdownCard>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'Breakdown',
    hourlyRate: { label: 'Hourly rate', value: '€0.45/hr' },
    hoursPerMonth: { label: 'Hours per month', value: '600h' },
    utilization: { label: 'Utilization', value: 80 },
    commission: { label: 'rbee commission (15%)', value: '-€32' },
    takeHome: { label: 'Your take-home', value: '€184' },
  },
}

export const HighUtilization: Story = {
  args: {
    title: 'Breakdown',
    hourlyRate: { label: 'Hourly rate', value: '€1.20/hr' },
    hoursPerMonth: { label: 'Hours per month', value: '720h' },
    utilization: { label: 'Utilization', value: 95 },
    commission: { label: 'rbee commission (15%)', value: '-€123' },
    takeHome: { label: 'Your take-home', value: '€697' },
  },
}

export const LowUtilization: Story = {
  args: {
    title: 'Breakdown',
    hourlyRate: { label: 'Hourly rate', value: '€0.35/hr' },
    hoursPerMonth: { label: 'Hours per month', value: '240h' },
    utilization: { label: 'Utilization', value: 30 },
    commission: { label: 'rbee commission (15%)', value: '-€4' },
    takeHome: { label: 'Your take-home', value: '€21' },
  },
}

export const InContainer: Story = {
  args: {
    title: 'Monthly Breakdown',
    hourlyRate: { label: 'Hourly rate', value: '€0.45/hr' },
    hoursPerMonth: { label: 'Hours per month', value: '600h' },
    utilization: { label: 'Utilization', value: 80 },
    commission: { label: 'rbee commission (15%)', value: '-€32' },
    takeHome: { label: 'Your take-home', value: '€184' },
  },
  render: (args) => (
    <div className="max-w-md">
      <EarningsBreakdownCard {...args} />
    </div>
  ),
}
