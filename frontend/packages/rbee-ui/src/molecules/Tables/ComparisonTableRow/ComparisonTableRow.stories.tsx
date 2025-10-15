import type { Meta, StoryObj } from '@storybook/react'
import { ComparisonTableRow } from './ComparisonTableRow'

const meta: Meta<typeof ComparisonTableRow> = {
	title: 'Molecules/Tables/ComparisonTableRow',
	component: ComparisonTableRow,
	parameters: {
		layout: 'padded',
		docs: {
			description: {
				component: `
## Overview
The ComparisonTableRow molecule displays a single row in a comparison table with feature name and values across columns.

## Used In
- Pricing comparison tables
- Feature comparison matrices
				`,
			},
		},
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ComparisonTableRow>

export const Default: Story = {
	render: () => (
		<table className="w-full"><tbody>
			<ComparisonTableRow feature="API Access" values={[true, true, true]} />
		</tbody></table>
	),
}

export const WithBooleans: Story = {
	render: () => (
		<table className="w-full"><tbody>
			<ComparisonTableRow feature="Multi-GPU" values={[false, true, true]} />
		</tbody></table>
	),
}

export const WithStrings: Story = {
	render: () => (
		<table className="w-full"><tbody>
			<ComparisonTableRow feature="Workers" values={['1', '10', 'Unlimited']} />
		</tbody></table>
	),
}

export const WithHighlight: Story = {
	render: () => (
		<table className="w-full"><tbody>
			<ComparisonTableRow feature="Support" values={['Community', 'Email', '24/7']} highlightColumn={1} />
		</tbody></table>
	),
}
