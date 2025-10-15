// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Check } from 'lucide-react'

const meta: Meta = {
  title: 'Atoms/CheckItem',
  parameters: { layout: 'centered' },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj

export const Default: Story = {
  render: () => (
    <div className="flex items-center gap-2 p-3 border rounded-lg">
      <div className="w-4 h-4 border rounded flex items-center justify-center" />
      <span className="text-sm">Check item</span>
    </div>
  ),
}

export const Checked: Story = {
  render: () => (
    <div className="flex items-center gap-2 p-3 border rounded-lg">
      <div className="w-4 h-4 border rounded flex items-center justify-center bg-primary text-primary-foreground">
        <Check className="w-3 h-3" />
      </div>
      <span className="text-sm">Checked item</span>
    </div>
  ),
}

export const WithDescription: Story = {
  render: () => (
    <div className="flex items-start gap-3 p-3 border rounded-lg max-w-md">
      <div className="w-4 h-4 border rounded flex items-center justify-center mt-0.5 bg-primary text-primary-foreground">
        <Check className="w-3 h-3" />
      </div>
      <div>
        <div className="text-sm font-medium">Check item with description</div>
        <div className="text-xs text-muted-foreground">This is a longer description explaining the check item</div>
      </div>
    </div>
  ),
}

export const Disabled: Story = {
  render: () => (
    <div className="flex items-center gap-2 p-3 border rounded-lg opacity-50">
      <div className="w-4 h-4 border rounded flex items-center justify-center" />
      <span className="text-sm">Disabled item</span>
    </div>
  ),
}
