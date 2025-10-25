// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'

const meta: Meta = {
  title: 'Atoms/Chart',
  parameters: { layout: 'centered' },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj

export const Default: Story = {
  render: () => (
    <div className="w-[400px] h-[300px] border rounded p-4 flex items-center justify-center bg-muted">
      <span className="text-sm text-muted-foreground">Chart component placeholder - requires charting library</span>
    </div>
  ),
}

export const AllTypes: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="w-[400px] h-[200px] border rounded p-4 flex items-center justify-center bg-muted">
        <span className="text-sm text-muted-foreground">Line Chart</span>
      </div>
      <div className="w-[400px] h-[200px] border rounded p-4 flex items-center justify-center bg-muted">
        <span className="text-sm text-muted-foreground">Bar Chart</span>
      </div>
    </div>
  ),
}

export const WithLegend: Story = {
  render: () => (
    <div className="w-[400px] border rounded p-4">
      <div className="h-[250px] flex items-center justify-center bg-muted rounded mb-4">
        <span className="text-sm text-muted-foreground">Chart Area</span>
      </div>
      <div className="flex gap-4 justify-center">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-blue-500 rounded" />
          <span className="text-xs">Series 1</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded" />
          <span className="text-xs">Series 2</span>
        </div>
      </div>
    </div>
  ),
}

export const Responsive: Story = {
  render: () => (
    <div className="w-full max-w-2xl h-[300px] border rounded p-4 flex items-center justify-center bg-muted">
      <span className="text-sm text-muted-foreground">Responsive Chart</span>
    </div>
  ),
}
