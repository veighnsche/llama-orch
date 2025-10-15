import type { Meta, StoryObj } from '@storybook/react'
import { GPUListItem } from './GPUListItem'

const meta: Meta<typeof GPUListItem> = {
  title: 'Molecules/GPUListItem',
  component: GPUListItem,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component: `
## Overview
The GPUListItem molecule displays GPU information with status indicator, name, and metrics.

## Used In
- GPU selection interfaces
- Resource monitoring dashboards
- Provider earnings pages
				`,
      },
    },
  },
  tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof GPUListItem>

export const Default: Story = {
  args: {
    name: 'NVIDIA RTX 4090',
    value: '24 GB',
    label: 'VRAM',
    status: 'active',
  },
}

export const WithSpecs: Story = {
  args: {
    name: 'NVIDIA A100',
    subtitle: '80GB PCIe',
    value: '€2.50',
    label: 'per hour',
    status: 'active',
  },
}

export const Idle: Story = {
  args: {
    name: 'NVIDIA RTX 3090',
    subtitle: '24GB GDDR6X',
    value: 'Idle',
    status: 'idle',
  },
}

export const Offline: Story = {
  args: {
    name: 'NVIDIA V100',
    subtitle: '16GB HBM2',
    value: 'Offline',
    status: 'offline',
  },
}
