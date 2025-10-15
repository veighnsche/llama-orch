import type { Meta, StoryObj } from '@storybook/react'
import { PledgeCallout } from './PledgeCallout'

const meta: Meta<typeof PledgeCallout> = {
	title: 'Molecules/Content/PledgeCallout',
	component: PledgeCallout,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The PledgeCallout molecule displays a security/trust message with icon and link.

## Used In
- Security pages
- Trust indicators
- Compliance sections
				`,
			},
		},
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof PledgeCallout>

export const Default: Story = {
	args: {},
}

export const WithIcon: Story = {
	args: {},
}

export const AllVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-4 p-8 max-w-2xl">
			<PledgeCallout />
		</div>
	),
}

export const WithAction: Story = {
	args: {},
}
