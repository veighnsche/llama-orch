import type { Meta, StoryObj } from '@storybook/react'
import { BrandLogo } from './BrandLogo'

const meta = {
  title: 'Molecules/BrandLogo',
  component: BrandLogo,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
      description: 'Size variant of the logo',
    },
    showWordmark: {
      control: 'boolean',
      description: 'Whether to show the "rbee" wordmark',
    },
    href: {
      control: 'text',
      description: 'Link destination (set to empty string to disable link)',
    },
    priority: {
      control: 'boolean',
      description: 'Whether to prioritize image loading',
    },
  },
} satisfies Meta<typeof BrandLogo>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    size: 'md',
    showWordmark: true,
    href: '/',
    priority: false,
  },
}

export const Small: Story = {
  args: {
    size: 'sm',
    showWordmark: true,
    href: '/',
  },
}

export const Large: Story = {
  args: {
    size: 'lg',
    showWordmark: true,
    href: '/',
  },
}

export const IconOnly: Story = {
  args: {
    size: 'md',
    showWordmark: false,
    href: '/',
  },
}

export const NoLink: Story = {
  args: {
    size: 'md',
    showWordmark: true,
    href: '',
  },
}

export const AllSizes: Story = {
  render: () => (
    <div className="flex flex-col gap-8 items-start">
      <div>
        <p className="text-sm text-muted-foreground mb-2">Small</p>
        <BrandLogo size="sm" />
      </div>
      <div>
        <p className="text-sm text-muted-foreground mb-2">Medium (Default)</p>
        <BrandLogo size="md" />
      </div>
      <div>
        <p className="text-sm text-muted-foreground mb-2">Large</p>
        <BrandLogo size="lg" />
      </div>
    </div>
  ),
}

export const IconOnlyVariants: Story = {
  render: () => (
    <div className="flex gap-6 items-center">
      <BrandLogo size="sm" showWordmark={false} />
      <BrandLogo size="md" showWordmark={false} />
      <BrandLogo size="lg" showWordmark={false} />
    </div>
  ),
}
