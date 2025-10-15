import type { Meta, StoryObj } from '@storybook/react'
import { StepNumber } from './StepNumber'

const meta: Meta<typeof StepNumber> = {
	title: 'Molecules/Content/StepNumber',
	component: StepNumber,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The StepNumber molecule displays a numbered badge for step indicators.

## Used In
- Step-by-step guides
- Process flows
- Numbered lists
				`,
			},
		},
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof StepNumber>

export const Default: Story = {
	args: { number: 1 },
}

export const Active: Story = {
	args: { number: 2, variant: 'primary' },
}

export const Completed: Story = {
	args: { number: 3, variant: 'secondary' },
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex items-center gap-4 p-8">
			<StepNumber number={1} size="sm" />
			<StepNumber number={2} size="md" />
			<StepNumber number={3} size="lg" />
			<StepNumber number={4} size="xl" />
		</div>
	),
}
