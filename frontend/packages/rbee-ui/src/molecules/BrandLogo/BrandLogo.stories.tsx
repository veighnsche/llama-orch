import type { Meta, StoryObj } from '@storybook/react'
import { BrandLogo } from './BrandLogo'

const meta: Meta<typeof BrandLogo> = {
	title: 'Molecules/BrandLogo',
	component: BrandLogo,
	parameters: {
		layout: 'centered',
		backgrounds: {
			default: 'dark',
		},
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
		href: {
			control: 'text',
		},
	},
}

export default meta
type Story = StoryObj<typeof BrandLogo>

export const Default: Story = {
	args: {
		size: 'md',
		href: '/',
	},
}

export const Small: Story = {
	args: {
		size: 'sm',
		href: '/',
	},
}

export const Large: Story = {
	args: {
		size: 'lg',
		href: '/',
	},
}

export const WithoutLink: Story = {
	args: {
		size: 'md',
		href: undefined,
	},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex flex-col gap-8 p-8">
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Small</h3>
				<BrandLogo size="sm" />
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Medium (Default)</h3>
				<BrandLogo size="md" />
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Large</h3>
				<BrandLogo size="lg" />
			</div>
		</div>
	),
}
