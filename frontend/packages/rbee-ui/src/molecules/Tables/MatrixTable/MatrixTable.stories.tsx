import type { Meta, StoryObj } from '@storybook/react'
import { MatrixTable } from './MatrixTable'

const meta: Meta<typeof MatrixTable> = {
	title: 'Molecules/Tables/MatrixTable',
	component: MatrixTable,
	parameters: {
		layout: 'padded',
		docs: {
			description: {
				component: `
## Overview
The MatrixTable molecule displays a comparison matrix with providers as columns and features as rows.

## Used In
- Enterprise compliance comparison
- Provider feature comparison
				`,
			},
		},
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof MatrixTable>

const columns = [
	{ key: 'rbee', label: 'rbee', accent: true },
	{ key: 'aws', label: 'AWS' },
	{ key: 'azure', label: 'Azure' },
]

const rows = [
	{ feature: 'GDPR Compliance', values: { rbee: true, aws: true, azure: true } },
	{ feature: 'Data Residency', values: { rbee: 'Netherlands', aws: 'Global', azure: 'EU' } },
	{ feature: 'SOC2', values: { rbee: 'Partial', aws: true, azure: true } },
]

export const Default: Story = {
	args: { columns, rows },
}

export const WithHeaders: Story = {
	args: { columns, rows },
}

export const WithColors: Story = {
	args: { columns, rows },
}

export const Sortable: Story = {
	args: { columns, rows },
}
