import type { Meta, StoryObj } from '@storybook/react'
import { ProgressBar } from './ProgressBar'

const meta: Meta<typeof ProgressBar> = {
	title: 'Molecules/ProgressBar',
	component: ProgressBar,
	parameters: {
		layout: 'padded',
		docs: {
			description: {
				component: `
## Overview
ProgressBar is a versatile progress indicator molecule used to show completion status, resource utilization, or loading states across the commercial site.

## Composition
This molecule is composed of:
- **Label**: Optional text label describing the progress
- **Track**: Background container for the progress bar
- **Fill**: Colored bar indicating progress percentage
- **Percentage**: Optional percentage text display

## When to Use
- Resource utilization (GPU memory, CPU usage)
- Download/upload progress
- Task completion status
- Loading states with known progress
- Multi-step processes

## Variants
- **Sizes**: sm, md, lg
- **Colors**: primary, chart-1 through chart-5
- **With/without label**: Optional label display
- **With/without percentage**: Optional percentage display

## Used In Commercial Site
Used in 2 organisms:
- ResourceMonitor (GPU/CPU utilization)
- DownloadProgress (model download status)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		label: {
			control: 'text',
			description: 'Progress label',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		percentage: {
			control: { type: 'range', min: 0, max: 100, step: 1 },
			description: 'Progress percentage (0-100)',
			table: {
				type: { summary: 'number' },
				category: 'Content',
			},
		},
		color: {
			control: 'select',
			options: ['primary', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
			description: 'Progress bar color (Tailwind class)',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'primary' },
				category: 'Appearance',
			},
		},
		size: {
			control: 'select',
			options: ['sm', 'md', 'lg'],
			description: 'Size variant',
			table: {
				type: { summary: "'sm' | 'md' | 'lg'" },
				defaultValue: { summary: 'md' },
				category: 'Appearance',
			},
		},
		showLabel: {
			control: 'boolean',
			description: 'Show label',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Appearance',
			},
		},
		showPercentage: {
			control: 'boolean',
			description: 'Show percentage',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof ProgressBar>

export const Default: Story = {
	args: {
		label: 'GPU Memory',
		percentage: 65,
		color: 'primary',
		size: 'md',
		showLabel: true,
		showPercentage: true,
	},
}

export const WithPercentage: Story = {
	render: () => (
		<div className="space-y-4 w-full max-w-md">
			<ProgressBar label="Download" percentage={25} color="chart-2" />
			<ProgressBar label="Processing" percentage={50} color="chart-3" />
			<ProgressBar label="Complete" percentage={100} color="chart-3" />
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Progress bars showing different completion states with percentages.',
			},
		},
	},
}

export const AllStates: Story = {
	render: () => (
		<div className="space-y-6 w-full max-w-md">
			<div>
				<h3 className="text-sm font-semibold mb-3">Sizes</h3>
				<div className="space-y-3">
					<ProgressBar label="Small" percentage={60} size="sm" />
					<ProgressBar label="Medium" percentage={60} size="md" />
					<ProgressBar label="Large" percentage={60} size="lg" />
				</div>
			</div>
			<div>
				<h3 className="text-sm font-semibold mb-3">Colors</h3>
				<div className="space-y-3">
					<ProgressBar label="Primary" percentage={70} color="primary" />
					<ProgressBar label="Chart 1" percentage={70} color="chart-1" />
					<ProgressBar label="Chart 2" percentage={70} color="chart-2" />
					<ProgressBar label="Chart 3" percentage={70} color="chart-3" />
				</div>
			</div>
			<div>
				<h3 className="text-sm font-semibold mb-3">Options</h3>
				<div className="space-y-3">
					<ProgressBar label="With all" percentage={80} showLabel showPercentage />
					<ProgressBar label="No label" percentage={80} showLabel={false} showPercentage />
					<ProgressBar label="No percentage" percentage={80} showLabel showPercentage={false} />
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available sizes, colors, and display options.',
			},
		},
	},
}

export const Animated: Story = {
	render: () => {
		const ResourceMonitor = () => {
			return (
				<div className="w-full max-w-2xl p-6 bg-card rounded-lg border">
					<h3 className="text-lg font-semibold mb-4">Resource Utilization</h3>
					<div className="space-y-4">
						<ProgressBar label="GPU 1 (RTX 4090)" percentage={85} color="chart-1" size="md" />
						<ProgressBar label="GPU 2 (RTX 3080)" percentage={62} color="chart-1" size="md" />
						<ProgressBar label="CPU Usage" percentage={34} color="chart-2" size="md" />
						<ProgressBar label="RAM Usage" percentage={58} color="chart-3" size="md" />
						<ProgressBar label="VRAM (GPU 1)" percentage={92} color="chart-4" size="md" />
						<ProgressBar label="VRAM (GPU 2)" percentage={71} color="chart-4" size="md" />
					</div>
				</div>
			)
		}

		return <ResourceMonitor />
	},
	parameters: {
		docs: {
			description: {
				story: 'ProgressBar as used in ResourceMonitor, showing real-time resource utilization.',
			},
		},
	},
}
