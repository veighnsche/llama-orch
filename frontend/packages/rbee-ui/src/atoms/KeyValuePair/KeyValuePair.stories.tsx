import type { Meta, StoryObj } from '@storybook/react'
import { KeyValuePair } from './KeyValuePair'

const meta = {
	title: 'Atoms/KeyValuePair',
	component: KeyValuePair,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		label: {
			control: 'text',
			description: 'The label/key text',
		},
		value: {
			control: 'text',
			description: 'The value text',
		},
		valueVariant: {
			control: 'select',
			options: ['default', 'semibold', 'bold', 'success', 'warning', 'error'],
			description: 'Style variant for the value',
		},
	},
} satisfies Meta<typeof KeyValuePair>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		label: 'GPU Pool',
		value: '5 nodes / 8 GPUs',
	},
}

export const Semibold: Story = {
	args: {
		label: 'GPU Pool',
		value: '5 nodes / 8 GPUs',
		valueVariant: 'semibold',
	},
}

export const Bold: Story = {
	args: {
		label: 'Status',
		value: 'Active',
		valueVariant: 'bold',
	},
}

export const Success: Story = {
	args: {
		label: 'Cost',
		value: '$0.00 / hr',
		valueVariant: 'success',
	},
}

export const Warning: Story = {
	args: {
		label: 'Temperature',
		value: '75°C',
		valueVariant: 'warning',
	},
}

export const Error: Story = {
	args: {
		label: 'Error Rate',
		value: '12%',
		valueVariant: 'error',
	},
}

export const LongText: Story = {
	args: {
		label: 'Configuration',
		value: 'NVIDIA A100 80GB PCIe (x8)',
		valueVariant: 'semibold',
	},
}
