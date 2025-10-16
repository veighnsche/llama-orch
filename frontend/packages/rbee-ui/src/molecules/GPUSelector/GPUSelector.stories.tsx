import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { GPUSelector, type GPUSelectorModel } from './GPUSelector'

const meta = {
  title: 'Molecules/GPUSelector',
  component: GPUSelector,
  parameters: {
    layout: 'padded',
  },
  tags: ['autodocs'],
} satisfies Meta<typeof GPUSelector>

export default meta
type Story = StoryObj<typeof meta>

const sampleGPUs: GPUSelectorModel[] = [
  { name: 'NVIDIA RTX 4090', baseRate: 1.2, vram: 24 },
  { name: 'NVIDIA RTX 4080', baseRate: 0.9, vram: 16 },
  { name: 'NVIDIA RTX 3090', baseRate: 0.8, vram: 24 },
  { name: 'NVIDIA A100', baseRate: 2.5, vram: 80 },
  { name: 'NVIDIA H100', baseRate: 3.8, vram: 80 },
  { name: 'AMD MI250X', baseRate: 2.2, vram: 128 },
]

const formatHourly = (rate: number) => `€${rate.toFixed(2)}/hr`

export const Default: Story = {
  render: () => {
    const [selected, setSelected] = useState(sampleGPUs[0])
    return (
      <div className="max-w-md">
        <GPUSelector
          models={sampleGPUs}
          selectedModel={selected}
          onSelect={setSelected}
          label="Select GPU Model"
          formatHourly={formatHourly}
        />
      </div>
    )
  },
}

export const FewOptions: Story = {
  render: () => {
    const [selected, setSelected] = useState(sampleGPUs[0])
    return (
      <div className="max-w-md">
        <GPUSelector
          models={sampleGPUs.slice(0, 3)}
          selectedModel={selected}
          onSelect={setSelected}
          label="Choose Your GPU"
          formatHourly={formatHourly}
        />
      </div>
    )
  },
}

export const ManyOptions: Story = {
  render: () => {
    const manyGPUs = [
      ...sampleGPUs,
      { name: 'NVIDIA RTX 3080', baseRate: 0.7, vram: 10 },
      { name: 'NVIDIA RTX 3070', baseRate: 0.5, vram: 8 },
      { name: 'AMD RX 7900 XTX', baseRate: 0.6, vram: 24 },
      { name: 'AMD RX 6900 XT', baseRate: 0.4, vram: 16 },
    ]
    const [selected, setSelected] = useState(manyGPUs[0])
    return (
      <div className="max-w-md">
        <GPUSelector
          models={manyGPUs}
          selectedModel={selected}
          onSelect={setSelected}
          label="Select GPU Model"
          formatHourly={formatHourly}
        />
      </div>
    )
  },
}
