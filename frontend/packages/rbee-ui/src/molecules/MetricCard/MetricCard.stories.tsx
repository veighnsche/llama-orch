import type { Meta, StoryObj } from '@storybook/react'
import { MetricCard } from './MetricCard'

const meta = {
  title: 'Molecules/MetricCard',
  component: MetricCard,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof MetricCard>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default metric card
 * Shows a simple label and value
 */
export const Default: Story = {
  args: {
    label: 'Active Workers',
    value: '12',
  },
}

/**
 * Currency value
 * Shows monetary metric
 */
export const Currency: Story = {
  args: {
    label: 'Monthly Revenue',
    value: '€1,234',
  },
}

/**
 * Percentage value
 * Shows percentage metric
 */
export const Percentage: Story = {
  args: {
    label: 'GPU Utilization',
    value: '87%',
  },
}

/**
 * Large number
 * Shows metric with large value
 */
export const LargeNumber: Story = {
  args: {
    label: 'Total Requests',
    value: '1,234,567',
  },
}

/**
 * Decimal value
 * Shows metric with decimal precision
 */
export const Decimal: Story = {
  args: {
    label: 'Avg Response Time',
    value: '2.3s',
  },
}

/**
 * Time duration
 * Shows time-based metric
 */
export const TimeDuration: Story = {
  args: {
    label: 'Uptime',
    value: '99.9%',
  },
}

/**
 * Multiple cards in grid
 * Shows how cards look together
 */
export const Grid = {
  render: () => (
    <div className="grid grid-cols-3 gap-4">
      <MetricCard label="Active Workers" value="12" />
      <MetricCard label="Total VRAM" value="192 GB" />
      <MetricCard label="Utilization" value="87%" />
      <MetricCard label="Requests/min" value="1,234" />
      <MetricCard label="Avg Latency" value="45ms" />
      <MetricCard label="Uptime" value="99.9%" />
    </div>
  ),
}
