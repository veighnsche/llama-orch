import type { Meta, StoryObj } from '@storybook/react'
import { Activity, CheckCircle, Clock, TrendingUp } from 'lucide-react'
import { StatusKPI } from './StatusKPI'

const meta: Meta<typeof StatusKPI> = {
	title: 'Molecules/StatusKPI',
	component: StatusKPI,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The StatusKPI molecule displays a key performance indicator with an icon, label, and value. Used for system status dashboards and monitoring displays.

## Composition
This molecule is composed of:
- **IconBox**: Status icon with color
- **Label**: Metric name
- **Value**: Metric value (string or number)

## When to Use
- System status dashboards
- Monitoring displays
- KPI cards
- Health check indicators
- Performance metrics

## Used In
- **ErrorHandling**: Displays system health KPIs (uptime, error rate, response time)
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			control: false,
			description: 'Lucide icon component',
			table: {
				type: { summary: 'LucideIcon' },
				category: 'Content',
			},
		},
		color: {
			control: 'select',
			options: ['primary', 'chart-2', 'chart-3', 'chart-4', 'destructive'],
			description: 'Icon color',
			table: {
				type: { summary: 'IconBoxProps["color"]' },
				category: 'Appearance',
			},
		},
		label: {
			control: 'text',
			description: 'Metric label',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		value: {
			control: 'text',
			description: 'Metric value',
			table: {
				type: { summary: 'string | number' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof StatusKPI>

export const Default: Story = {
	args: {
		icon: Activity,
		color: 'chart-3',
		label: 'Uptime',
		value: '99.9%',
	},
}

export const AllStates: Story = {
	render: () => (
		<div className="flex flex-col gap-4 p-8">
			<h3 className="text-lg font-semibold text-foreground mb-2">All Status States</h3>
			<div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
				<StatusKPI icon={CheckCircle} color="chart-3" label="Uptime" value="99.9%" />
				<StatusKPI icon={Activity} color="primary" label="Active Workers" value="12" />
				<StatusKPI icon={Clock} color="chart-2" label="Avg Response" value="45ms" />
				<StatusKPI icon={TrendingUp} color="chart-4" label="Requests/min" value="1.2k" />
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available status states with different icons and colors.',
			},
		},
	},
}

export const WithTrend: Story = {
	render: () => (
		<div className="flex flex-col gap-4 p-8">
			<h3 className="text-lg font-semibold text-foreground mb-2">KPIs with Trend Indicators</h3>
			<div className="grid gap-4 sm:grid-cols-3">
				<div className="space-y-2">
					<StatusKPI icon={Activity} color="chart-3" label="Uptime" value="99.9%" />
					<div className="text-xs text-chart-3 flex items-center gap-1 pl-4">
						<TrendingUp className="h-3 w-3" />
						<span>+0.1% from last week</span>
					</div>
				</div>
				<div className="space-y-2">
					<StatusKPI icon={Clock} color="primary" label="Response Time" value="42ms" />
					<div className="text-xs text-chart-3 flex items-center gap-1 pl-4">
						<TrendingUp className="h-3 w-3" />
						<span>-8ms improvement</span>
					</div>
				</div>
				<div className="space-y-2">
					<StatusKPI icon={CheckCircle} color="chart-2" label="Success Rate" value="99.8%" />
					<div className="text-xs text-muted-foreground flex items-center gap-1 pl-4">
						<span>Stable</span>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'StatusKPI with trend indicators showing improvements and stability.',
			},
		},
	},
}

export const InErrorHandlingContext: Story = {
	render: () => (
		<div className="w-full max-w-5xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: StatusKPI in ErrorHandling organism
			</div>
			<div className="rounded-2xl border bg-card p-6">
				<h3 className="text-xl font-semibold text-foreground mb-4">System Health</h3>
				<div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
					<StatusKPI icon={CheckCircle} color="chart-3" label="Uptime" value="99.9%" />
					<StatusKPI icon={Activity} color="primary" label="Active Workers" value="12" />
					<StatusKPI icon={Clock} color="chart-2" label="Avg Response" value="45ms" />
					<StatusKPI icon={TrendingUp} color="chart-4" label="Requests/min" value="1.2k" />
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'StatusKPI as used in the ErrorHandling organism, displaying system health metrics in a grid.',
			},
		},
	},
}
