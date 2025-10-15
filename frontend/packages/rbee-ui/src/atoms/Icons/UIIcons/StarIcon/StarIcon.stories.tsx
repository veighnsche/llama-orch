import type { Meta, StoryObj } from '@storybook/react'
import { StarIcon } from './StarIcon'

const meta: Meta<typeof StarIcon> = {
  title: 'Atoms/Icons/UI/StarIcon',
  component: StarIcon,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    filled: {
      control: 'boolean',
      description: 'Whether the star is filled',
    },
  },
}

export default meta
type Story = StoryObj<typeof StarIcon>

export const Default: Story = {
  args: {
    filled: false,
  },
}

export const Filled: Story = {
  args: {
    filled: true,
  },
}

export const HalfFilled: Story = {
  render: () => (
    <div className="flex gap-1">
      <StarIcon filled />
      <StarIcon filled />
      <StarIcon filled />
      <div className="relative">
        <StarIcon filled={false} />
        <div className="absolute inset-0 overflow-hidden" style={{ width: '50%' }}>
          <StarIcon filled />
        </div>
      </div>
      <StarIcon filled={false} />
    </div>
  ),
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-2">
        <StarIcon filled className="size-3" />
        <span className="text-sm">Small (12px)</span>
      </div>
      <div className="flex items-center gap-2">
        <StarIcon filled className="size-4" />
        <span className="text-sm">Default (16px)</span>
      </div>
      <div className="flex items-center gap-2">
        <StarIcon filled className="size-5" />
        <span className="text-sm">Medium (20px)</span>
      </div>
      <div className="flex items-center gap-2">
        <StarIcon filled className="size-6" />
        <span className="text-sm">Large (24px)</span>
      </div>
      <div className="flex items-center gap-2">
        <StarIcon filled className="size-8" />
        <span className="text-sm">Extra Large (32px)</span>
      </div>
    </div>
  ),
}
