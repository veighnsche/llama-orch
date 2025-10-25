import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { OptionSelector } from './OptionSelector'

const meta = {
  title: 'Molecules/OptionSelector',
  component: OptionSelector,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof OptionSelector>

export default meta
type Story = StoryObj<typeof meta>

const earningsPresets = [
  {
    id: 'casual',
    label: 'Casual',
    subtitle: '8h • 50%',
    data: { hours: 8, utilization: 50 },
  },
  {
    id: 'daily',
    label: 'Daily',
    subtitle: '16h • 70%',
    data: { hours: 16, utilization: 70 },
  },
  {
    id: 'always-on',
    label: 'Always On',
    subtitle: '24h • 90%',
    data: { hours: 24, utilization: 90 },
  },
]

export const Default: Story = {
  args: {} as any,
  render: () => {
    const [selected, setSelected] = useState<any>(null)
    return (
      <div className="max-w-md space-y-4">
        <div>
          <label className="mb-3 block text-sm font-medium text-muted-foreground">Quick Presets</label>
          <OptionSelector options={earningsPresets} onSelect={setSelected} columns={3} />
        </div>
        {selected && (
          <div className="rounded border bg-muted p-4 text-sm">
            <strong>Selected:</strong> {selected.hours}h/day at {selected.utilization}% utilization
          </div>
        )}
      </div>
    )
  },
}

export const TwoColumns: Story = {
  args: {} as any,
  render: () => (
    <div className="max-w-md">
      <div>
        <label className="mb-3 block text-sm font-medium text-muted-foreground">Choose Plan</label>
        <OptionSelector
          options={[
            {
              id: 'basic',
              label: 'Basic',
              subtitle: '€9/month',
              data: { plan: 'basic', price: 9 },
            },
            {
              id: 'pro',
              label: 'Pro',
              subtitle: '€29/month',
              data: { plan: 'pro', price: 29 },
            },
          ]}
          onSelect={(data) => console.log(data)}
          columns={2}
        />
      </div>
    </div>
  ),
}

export const FourColumns: Story = {
  args: {} as any,
  render: () => (
    <div className="max-w-2xl">
      <div>
        <label className="mb-3 block text-sm font-medium text-muted-foreground">Time Range</label>
        <OptionSelector
          options={[
            { id: '1h', label: '1 Hour', data: { hours: 1 } },
            { id: '24h', label: '24 Hours', data: { hours: 24 } },
            { id: '7d', label: '7 Days', data: { days: 7 } },
            { id: '30d', label: '30 Days', data: { days: 30 } },
          ]}
          onSelect={(data) => console.log(data)}
          columns={4}
        />
      </div>
    </div>
  ),
}

export const NoLabel: Story = {
  args: {} as any,
  render: () => (
    <div className="max-w-md">
      <OptionSelector options={earningsPresets} onSelect={(data) => console.log(data)} columns={3} />
    </div>
  ),
}

export const WithoutSubtitles: Story = {
  args: {} as any,
  render: () => (
    <div className="max-w-md">
      <div>
        <label className="mb-3 block text-sm font-medium text-muted-foreground">Select Mode</label>
        <OptionSelector
          options={[
            { id: 'light', label: 'Light', data: { theme: 'light' } },
            { id: 'dark', label: 'Dark', data: { theme: 'dark' } },
            { id: 'auto', label: 'Auto', data: { theme: 'auto' } },
          ]}
          onSelect={(data) => console.log(data)}
          columns={3}
        />
      </div>
    </div>
  ),
}
