import type { Meta, StoryObj } from '@storybook/react'
import { Code, Server, Shield } from 'lucide-react'
import { TabButton } from './TabButton'

const meta: Meta<typeof TabButton> = {
	title: 'Molecules/Navigation/TabButton',
	component: TabButton,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The TabButton molecule displays a tab button with icon and label.

## Used In
- Tab navigation
- Feature toggles
- View switchers
				`,
			},
		},
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof TabButton>

export const Default: Story = {
	args: {
		id: 'api',
		label: 'API',
		icon: Code,
		active: false,
		onClick: () => {},
	},
}

export const Active: Story = {
	args: {
		id: 'deployment',
		label: 'Deployment',
		icon: Server,
		active: true,
		onClick: () => {},
	},
}

export const WithIcon: Story = {
	args: {
		id: 'security',
		label: 'Security',
		icon: Shield,
		active: false,
		onClick: () => {},
	},
}

export const Disabled: Story = {
	render: () => (
		<div className="flex gap-2 p-8">
			<TabButton id="tab1" label="Available" icon={Code} active={false} onClick={() => {}} />
			<TabButton id="tab2" label="Active" icon={Server} active={true} onClick={() => {}} />
		</div>
	),
}
