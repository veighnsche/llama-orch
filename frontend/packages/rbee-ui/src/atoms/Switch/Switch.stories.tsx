import type { Meta, StoryObj } from '@storybook/react'
import { Switch } from './Switch'
import { Label } from '@rbee/ui/atoms/Label'

const meta: Meta<typeof Switch> = {
	title: 'Atoms/Switch',
	component: Switch,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		disabled: {
			control: 'boolean',
			description: 'Whether the switch is disabled',
		},
	},
}

export default meta
type Story = StoryObj<typeof Switch>

export const Default: Story = {
	args: {},
}

export const On: Story = {
	args: {
		defaultChecked: true,
	},
}

export const Disabled: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<Switch disabled />
			<Switch disabled defaultChecked />
		</div>
	),
}

export const WithLabel: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<div className="flex items-center gap-2">
				<Switch id="airplane-mode" />
				<Label htmlFor="airplane-mode">Airplane Mode</Label>
			</div>
			<div className="flex items-center gap-2">
				<Switch id="notifications" defaultChecked />
				<Label htmlFor="notifications">Enable Notifications</Label>
			</div>
			<div className="flex items-center gap-2">
				<Switch id="marketing" />
				<Label htmlFor="marketing">Marketing Emails</Label>
			</div>
			<div className="flex items-center gap-2">
				<Switch id="disabled-switch" disabled />
				<Label htmlFor="disabled-switch" className="opacity-50">
					Disabled Option
				</Label>
			</div>
		</div>
	),
}
