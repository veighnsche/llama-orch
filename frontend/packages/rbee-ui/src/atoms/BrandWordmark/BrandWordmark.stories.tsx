import type { Meta, StoryObj } from '@storybook/react'
import { BrandWordmark } from './BrandWordmark'

const meta: Meta<typeof BrandWordmark> = {
  title: 'Atoms/BrandWordmark',
  component: BrandWordmark,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg', 'xl', '2xl', '3xl', '4xl', '5xl'],
      description: 'Size of the wordmark',
    },
    inline: {
      control: 'boolean',
      description: 'Display inline with text',
    },
  },
}

export default meta
type Story = StoryObj<typeof BrandWordmark>

export const Default: Story = {
  args: {
    size: 'md',
  },
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4">
        <BrandWordmark size="sm" />
        <span className="text-sm text-muted-foreground">Small</span>
      </div>
      <div className="flex items-center gap-4">
        <BrandWordmark size="md" />
        <span className="text-sm text-muted-foreground">Medium (default)</span>
      </div>
      <div className="flex items-center gap-4">
        <BrandWordmark size="lg" />
        <span className="text-sm text-muted-foreground">Large</span>
      </div>
      <div className="flex items-center gap-4">
        <BrandWordmark size="xl" />
        <span className="text-sm text-muted-foreground">Extra Large</span>
      </div>
      <div className="flex items-center gap-4">
        <BrandWordmark size="2xl" />
        <span className="text-sm text-muted-foreground">2XL</span>
      </div>
      <div className="flex items-center gap-4">
        <BrandWordmark size="3xl" />
        <span className="text-sm text-muted-foreground">3XL</span>
      </div>
      <div className="flex items-center gap-4">
        <BrandWordmark size="4xl" />
        <span className="text-sm text-muted-foreground">4XL</span>
      </div>
      <div className="flex items-center gap-4">
        <BrandWordmark size="5xl" />
        <span className="text-sm text-muted-foreground">5XL</span>
      </div>
    </div>
  ),
}

export const WithColor: Story = {
  render: () => (
    <div className="flex flex-col gap-4">
      <BrandWordmark size="2xl" />
      <BrandWordmark size="2xl" className="text-primary" />
      <BrandWordmark size="2xl" className="text-blue-600" />
      <BrandWordmark size="2xl" className="text-green-600" />
      <BrandWordmark size="2xl" className="text-purple-600" />
    </div>
  ),
}

export const InBrandLogo: Story = {
  render: () => (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-2">
        <div className="flex size-10 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <span className="text-lg font-bold">R</span>
        </div>
        <BrandWordmark size="xl" />
      </div>
      <div className="flex items-center gap-2">
        <div className="flex size-12 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <span className="text-xl font-bold">R</span>
        </div>
        <BrandWordmark size="2xl" />
      </div>
      <div className="flex items-center gap-3">
        <div className="flex size-16 items-center justify-center rounded bg-primary text-primary-foreground">
          <span className="text-2xl font-bold">R</span>
        </div>
        <BrandWordmark size="3xl" />
      </div>
    </div>
  ),
}
