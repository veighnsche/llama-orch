import type { Meta, StoryObj } from '@storybook/react'
import { EarningsCard } from './EarningsCard'

const meta: Meta<typeof EarningsCard> = {
	title: 'Molecules/Providers/EarningsCard',
	component: EarningsCard,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The EarningsCard molecule displays earnings information with highlighted amount, stats grid, and optional breakdown. Used for provider earnings displays.

## Composition
This molecule is composed of:
- **Title**: Card title
- **Amount**: Large highlighted earnings amount
- **Subtitle**: Optional subtitle
- **Stats**: Grid of statistics (2 columns)
- **Breakdown**: Optional detailed breakdown list

## When to Use
- Provider earnings displays
- Revenue dashboards
- Financial summaries
- Performance metrics
- Payout information

## Used In
- **ProvidersEarnings**: Displays potential earnings for GPU providers
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		title: {
			control: 'text',
			description: 'Card title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		amount: {
			control: 'text',
			description: 'Main earnings amount',
			table: {
				type: { summary: 'number | string' },
				category: 'Content',
			},
		},
		subtitle: {
			control: 'text',
			description: 'Optional subtitle',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		stats: {
			control: 'object',
			description: 'Array of statistics',
			table: {
				type: { summary: 'EarningsStat[]' },
				category: 'Content',
			},
		},
		breakdown: {
			control: 'object',
			description: 'Optional breakdown items',
			table: {
				type: { summary: 'EarningsStat[]' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof EarningsCard>

export const Default: Story = {
	args: {
		title: 'Potential Monthly Earnings',
		amount: '2,450',
		subtitle: 'Based on 90% uptime',
		stats: [
			{ label: 'Hourly Rate', value: '€3.40' },
			{ label: 'Daily Earnings', value: '€81.60' },
		],
	},
}

export const WithBreakdown: Story = {
	args: {
		title: 'Potential Monthly Earnings',
		amount: '2,450',
		subtitle: 'Based on 90% uptime with RTX 4090',
		stats: [
			{ label: 'Hourly Rate', value: '€3.40' },
			{ label: 'Daily Earnings', value: '€81.60' },
			{ label: 'Uptime', value: '90%' },
			{ label: 'Requests', value: '12.5k' },
		],
		breakdown: [
			{ label: 'Base rate', value: '€2,100' },
			{ label: 'Performance bonus', value: '€250' },
			{ label: 'High availability', value: '€100' },
		],
	},
}

export const WithTrend: Story = {
	render: () => (
		<div className="flex flex-col gap-6 p-8 max-w-2xl">
			<h3 className="text-lg font-semibold text-foreground">Earnings with Trend</h3>
			<EarningsCard
				title="Your Earnings This Month"
				amount="2,680"
				subtitle="↑ 12% from last month"
				stats={[
					{ label: 'Hourly Rate', value: '€3.70' },
					{ label: 'Daily Earnings', value: '€88.80' },
					{ label: 'Uptime', value: '95%' },
					{ label: 'Requests', value: '15.2k' },
				]}
				breakdown={[
					{ label: 'Base rate', value: '€2,220' },
					{ label: 'Performance bonus', value: '€320' },
					{ label: 'High availability', value: '€140' },
				]}
			/>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'EarningsCard with trend indicator showing month-over-month growth.',
			},
		},
	},
}

export const InProvidersContext: Story = {
	render: () => (
		<div className="w-full max-w-5xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: EarningsCard in ProvidersEarnings organism
			</div>
			<div className="rounded-2xl border border-border bg-background p-8">
				<div className="mb-6">
					<h2 className="text-3xl font-bold text-foreground mb-2">Earn with Your GPU</h2>
					<p className="text-muted-foreground">
						Turn your idle GPU into a revenue stream. Here's what you could earn with an RTX 4090.
					</p>
				</div>
				<div className="grid gap-6 lg:grid-cols-2">
					<EarningsCard
						title="Conservative Estimate"
						amount="2,100"
						subtitle="70% uptime, standard workload"
						stats={[
							{ label: 'Hourly Rate', value: '€3.00' },
							{ label: 'Daily Earnings', value: '€70.00' },
							{ label: 'Uptime', value: '70%' },
							{ label: 'Requests', value: '8.5k' },
						]}
					/>
					<EarningsCard
						title="Optimistic Estimate"
						amount="3,200"
						subtitle="95% uptime, high demand"
						stats={[
							{ label: 'Hourly Rate', value: '€4.50' },
							{ label: 'Daily Earnings', value: '€106.50' },
							{ label: 'Uptime', value: '95%' },
							{ label: 'Requests', value: '18.2k' },
						]}
						breakdown={[
							{ label: 'Base rate', value: '€2,700' },
							{ label: 'Performance bonus', value: '€350' },
							{ label: 'High availability', value: '€150' },
						]}
					/>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'EarningsCard as used in the ProvidersEarnings organism, showing conservative and optimistic earnings scenarios.',
			},
		},
	},
}
