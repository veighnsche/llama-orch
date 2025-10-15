import type { Meta, StoryObj } from '@storybook/react'
import { Toggle } from './Toggle'
import { Bold, Italic, Underline } from 'lucide-react'

const meta: Meta<typeof Toggle> = {
	title: 'Atoms/Toggle',
	component: Toggle,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		variant: {
			control: 'select',
			options: ['default', 'outline'],
		},
		size: {
			control: 'select',
			options: ['default', 'sm', 'lg'],
		},
	},
}

export default meta
type Story = StoryObj<typeof Toggle>

export const Default: Story = {
	args: {
		children: 'Toggle',
	},
}

export const WithIcon: Story = {
	render: () => (
		<div className="flex gap-2">
			<Toggle aria-label="Toggle bold">
				<Bold className="size-4" />
			</Toggle>
			<Toggle aria-label="Toggle italic">
				<Italic className="size-4" />
			</Toggle>
			<Toggle aria-label="Toggle underline">
				<Underline className="size-4" />
			</Toggle>
		</div>
	),
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<div className="flex items-center gap-2">
				<Toggle size="sm">
					<Bold className="size-4" />
				</Toggle>
				<span className="text-sm">Small</span>
			</div>
			<div className="flex items-center gap-2">
				<Toggle size="default">
					<Bold className="size-4" />
				</Toggle>
				<span className="text-sm">Default</span>
			</div>
			<div className="flex items-center gap-2">
				<Toggle size="lg">
					<Bold className="size-4" />
				</Toggle>
				<span className="text-sm">Large</span>
			</div>
		</div>
	),
}

export const Disabled: Story = {
	render: () => (
		<div className="flex gap-2">
			<Toggle disabled>
				<Bold className="size-4" />
			</Toggle>
			<Toggle disabled defaultPressed>
				<Italic className="size-4" />
			</Toggle>
		</div>
	),
}
