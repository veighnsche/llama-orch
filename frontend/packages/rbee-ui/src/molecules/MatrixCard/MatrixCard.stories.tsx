import type { Meta, StoryObj } from '@storybook/react'
import { MatrixCard } from './MatrixCard'

const meta: Meta<typeof MatrixCard> = {
	title: 'Molecules/MatrixCard',
	component: MatrixCard,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The MatrixCard molecule displays a provider's features in a card format for mobile comparison views.

## Used In
- Mobile comparison tables
- Provider feature lists
				`,
			},
		},
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof MatrixCard>

const sampleProvider = { key: 'rbee', label: 'rbee', accent: true }
const sampleRows = [
	{ feature: 'GDPR Compliance', values: { rbee: true } },
	{ feature: 'Data Residency', values: { rbee: 'Netherlands' } },
	{ feature: 'SOC2', values: { rbee: 'Partial' } },
]

export const Default: Story = {
	args: {
		provider: sampleProvider,
		rows: sampleRows,
	},
}

export const WithData: Story = {
	args: {
		provider: { key: 'aws', label: 'AWS' },
		rows: [
			{ feature: 'GDPR', values: { aws: true } },
			{ feature: 'Location', values: { aws: 'Global' } },
		],
	},
}

export const WithColors: Story = {
	args: {
		provider: { key: 'rbee', label: 'rbee', accent: true },
		rows: sampleRows,
	},
}

export const Interactive: Story = {
	args: {
		provider: sampleProvider,
		rows: sampleRows,
	},
}
