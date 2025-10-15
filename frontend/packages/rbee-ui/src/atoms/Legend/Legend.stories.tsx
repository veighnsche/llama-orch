import type { Meta, StoryObj } from '@storybook/react'
import { Check, Circle, X } from 'lucide-react'
import { Legend } from './Legend'

const meta: Meta<typeof Legend> = {
  title: 'Atoms/Legend',
  component: Legend,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Legend>

export const Default: Story = {
  render: () => <Legend />,
}

export const WithColors: Story = {
  render: () => (
    <div className="flex flex-col gap-8 w-full max-w-2xl">
      <div className="flex flex-wrap items-center justify-center gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <Circle className="h-3 w-3 fill-blue-500 text-blue-500" />
          <span>Primary</span>
        </span>
        <span className="text-muted-foreground/40">·</span>
        <span className="flex items-center gap-1.5">
          <Circle className="h-3 w-3 fill-green-500 text-green-500" />
          <span>Success</span>
        </span>
        <span className="text-muted-foreground/40">·</span>
        <span className="flex items-center gap-1.5">
          <Circle className="h-3 w-3 fill-yellow-500 text-yellow-500" />
          <span>Warning</span>
        </span>
        <span className="text-muted-foreground/40">·</span>
        <span className="flex items-center gap-1.5">
          <Circle className="h-3 w-3 fill-red-500 text-red-500" />
          <span>Error</span>
        </span>
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded-sm bg-chart-1" />
          <span>Revenue</span>
        </span>
        <span className="text-muted-foreground/40">·</span>
        <span className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded-sm bg-chart-2" />
          <span>Expenses</span>
        </span>
        <span className="text-muted-foreground/40">·</span>
        <span className="flex items-center gap-1.5">
          <div className="h-3 w-3 rounded-sm bg-chart-3" />
          <span>Profit</span>
        </span>
      </div>
    </div>
  ),
}

export const Horizontal: Story = {
  render: () => (
    <div className="flex items-center gap-3 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <Check className="h-3.5 w-3.5 text-chart-3" />
        <span>Available</span>
      </span>
      <span className="text-muted-foreground/40">·</span>
      <span className="flex items-center gap-1.5">
        <X className="h-3.5 w-3.5 text-destructive" />
        <span>Unavailable</span>
      </span>
    </div>
  ),
}

export const Interactive: Story = {
  render: () => (
    <div className="flex flex-col gap-4 w-full max-w-md">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <button className="flex items-center gap-1.5 hover:text-foreground transition-colors">
          <Circle className="h-3 w-3 fill-blue-500 text-blue-500" />
          <span>Dataset A</span>
        </button>
        <span className="text-muted-foreground/40">·</span>
        <button className="flex items-center gap-1.5 hover:text-foreground transition-colors">
          <Circle className="h-3 w-3 fill-green-500 text-green-500" />
          <span>Dataset B</span>
        </button>
        <span className="text-muted-foreground/40">·</span>
        <button className="flex items-center gap-1.5 hover:text-foreground transition-colors">
          <Circle className="h-3 w-3 fill-yellow-500 text-yellow-500" />
          <span>Dataset C</span>
        </button>
      </div>
      <p className="text-xs text-muted-foreground">Click on legend items to toggle visibility</p>
    </div>
  ),
}
