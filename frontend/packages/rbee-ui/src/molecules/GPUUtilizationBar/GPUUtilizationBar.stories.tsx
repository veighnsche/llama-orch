import type { Meta, StoryObj } from '@storybook/react'
import { GPUUtilizationBar } from './GPUUtilizationBar'

const meta = {
	title: 'Molecules/GPUUtilizationBar',
	component: GPUUtilizationBar,
	parameters: {
		layout: 'padded',
	},
	tags: ['autodocs'],
	argTypes: {
		percentage: {
			control: { type: 'range', min: 0, max: 100, step: 1 },
		},
		variant: {
			control: 'select',
			options: ['primary', 'secondary'],
		},
	},
} satisfies Meta<typeof GPUUtilizationBar>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		label: 'RTX 4090',
		percentage: 75,
	},
}

export const HighUtilization: Story = {
	args: {
		label: 'RTX 4090 #1',
		percentage: 92,
	},
}

export const MediumUtilization: Story = {
	args: {
		label: 'M2 Ultra',
		percentage: 76,
	},
}

export const LowUtilization: Story = {
	args: {
		label: 'CPU Backend',
		percentage: 34,
		variant: 'secondary',
	},
}

export const ZeroUtilization: Story = {
	args: {
		label: 'Idle GPU',
		percentage: 0,
	},
}

export const FullUtilization: Story = {
	args: {
		label: 'Maxed Out',
		percentage: 100,
	},
}

export const SecondaryVariant: Story = {
	args: {
		label: 'CPU Backend',
		percentage: 45,
		variant: 'secondary',
	},
}

export const MultipleGPUs: Story = {
	args: {
		label: 'RTX 4090 #1',
		percentage: 92,
	},
	decorators: [
		() => (
			<div className="space-y-3 max-w-2xl">
				<GPUUtilizationBar label="RTX 4090 #1" percentage={92} />
				<GPUUtilizationBar label="RTX 4090 #2" percentage={88} />
				<GPUUtilizationBar label="M2 Ultra" percentage={76} />
				<GPUUtilizationBar label="CPU Backend" percentage={34} variant="secondary" />
			</div>
		),
	],
	parameters: {
		docs: {
			description: {
				story: 'Example showing multiple GPU utilization bars stacked vertically.',
			},
		},
	},
}
