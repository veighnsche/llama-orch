import type { Meta, StoryObj } from '@storybook/react'
import { BrandMark } from './BrandMark'

const meta: Meta<typeof BrandMark> = {
	title: 'Atoms/BrandMark',
	component: BrandMark,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		size: {
			control: 'select',
			options: ['sm', 'md', 'lg'],
		},
		priority: {
			control: 'boolean',
		},
	},
}

export default meta
type Story = StoryObj<typeof BrandMark>

export const Default: Story = {
	args: {
		size: 'md',
	},
}

export const Small: Story = {
	args: {
		size: 'sm',
	},
}

export const Large: Story = {
	args: {
		size: 'lg',
	},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex items-center gap-8">
			<div className="flex flex-col items-center gap-2">
				<BrandMark size="sm" />
				<span className="text-sm text-muted-foreground">Small</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<BrandMark size="md" />
				<span className="text-sm text-muted-foreground">Medium</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<BrandMark size="lg" />
				<span className="text-sm text-muted-foreground">Large</span>
			</div>
		</div>
	),
}
