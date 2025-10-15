import type { Meta, StoryObj } from '@storybook/react'
import { FloatingKPICard } from './FloatingKPICard'

const meta: Meta<typeof FloatingKPICard> = {
	title: 'Molecules/FloatingKPICard',
	component: FloatingKPICard,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
FloatingKPICard is a floating overlay molecule that displays key performance indicators (KPIs) with a glassmorphism effect. It's used to highlight real-time metrics in hero sections.

## Composition
This molecule is composed of:
- **Container**: Floating card with backdrop blur
- **KPI rows**: Label + value pairs
- **Animation**: Fade-in entrance animation
- **Glassmorphism**: Semi-transparent background with blur

## When to Use
- Hero sections (highlighting key metrics)
- Dashboard overlays (real-time stats)
- Feature demonstrations (showing performance)
- Visual emphasis (drawing attention to numbers)

## Variants
- Currently displays fixed KPIs (GPU Pool, Cost, Latency)
- Future: Configurable KPI data

## Used In Commercial Site
Used in:
- HeroSection (floating over hero image/diagram)
- DevelopersHero (showing performance metrics)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		className: {
			control: 'text',
			description: 'Additional CSS classes',
			table: {
				type: { summary: 'string' },
				category: 'Styling',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof FloatingKPICard>

export const Default: Story = {
	args: {},
}

export const WithTrend: Story = {
	render: () => (
		<div className="relative w-96 h-64 bg-gradient-to-br from-primary/20 to-chart-2/20 rounded-lg flex items-center justify-center">
			<div className="text-center">
				<h3 className="text-2xl font-bold mb-2">Your Infrastructure</h3>
				<p className="text-muted-foreground">Real-time metrics</p>
			</div>
			<FloatingKPICard />
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'FloatingKPICard positioned over a background element.',
			},
		},
	},
}

export const AllVariants: Story = {
	render: () => (
		<div className="space-y-8">
			<div>
				<h3 className="text-lg font-semibold mb-4">Light Background</h3>
				<div className="relative w-full h-64 bg-muted rounded-lg flex items-center justify-center">
					<div className="text-center">
						<h3 className="text-2xl font-bold">GPU Pool</h3>
					</div>
					<FloatingKPICard />
				</div>
			</div>
			<div>
				<h3 className="text-lg font-semibold mb-4">Gradient Background</h3>
				<div className="relative w-full h-64 bg-gradient-to-br from-chart-1/30 to-chart-3/30 rounded-lg flex items-center justify-center">
					<div className="text-center">
						<h3 className="text-2xl font-bold">Performance</h3>
					</div>
					<FloatingKPICard />
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'FloatingKPICard on different background types.',
			},
		},
	},
}

export const InHeroContext: Story = {
	render: () => (
		<div className="w-full max-w-4xl">
			<div className="relative bg-gradient-to-br from-primary/10 via-chart-2/10 to-chart-3/10 rounded-2xl p-12 overflow-hidden">
				<div className="relative z-10 text-center mb-8">
					<h1 className="text-4xl md:text-5xl font-bold mb-4">Private LLM Hosting</h1>
					<p className="text-lg text-muted-foreground mb-6">
						GDPR-compliant AI infrastructure in the Netherlands
					</p>
					<div className="flex justify-center gap-4">
						<button className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-colors">
							Get Started
						</button>
						<button className="px-6 py-3 bg-secondary text-secondary-foreground rounded-lg font-semibold hover:bg-secondary/80 transition-colors">
							Learn More
						</button>
					</div>
				</div>
				<div className="relative h-48 flex items-center justify-center">
					<div className="w-64 h-32 bg-card/40 rounded-lg border border-border flex items-center justify-center">
						<span className="text-muted-foreground">Architecture Diagram</span>
					</div>
					<FloatingKPICard />
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'FloatingKPICard as used in HeroSection, floating over architecture diagram.',
			},
		},
	},
}
