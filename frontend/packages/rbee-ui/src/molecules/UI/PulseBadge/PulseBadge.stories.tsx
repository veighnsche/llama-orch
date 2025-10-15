import type { Meta, StoryObj } from '@storybook/react'
import { PulseBadge } from './PulseBadge'

const meta: Meta<typeof PulseBadge> = {
	title: 'Molecules/UI/PulseBadge',
	component: PulseBadge,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
PulseBadge is an attention-grabbing badge molecule with an optional pulse animation. It's used to highlight new features, live status, or important announcements.

## Composition
This molecule is composed of:
- **Container**: Rounded pill with border and background
- **Pulse dot**: Animated dot indicator (optional)
- **Text**: Badge label

## When to Use
- New feature announcements (eyebrow badges)
- Live status indicators (streaming, online)
- Important notifications (alerts, updates)
- Beta/Alpha labels (product stages)
- Real-time status (active, processing)

## Variants
- **Colors**: primary, success, warning, info
- **Sizes**: sm, md, lg
- **Animation**: animated (pulse) or static

## Used In Commercial Site
Used in:
- HeroSection (new feature announcements)
- FeatureCards (beta labels)
- StatusIndicators (live status)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		text: {
			control: 'text',
			description: 'Badge text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		variant: {
			control: 'select',
			options: ['primary', 'success', 'warning', 'info'],
			description: 'Color variant',
			table: {
				type: { summary: "'primary' | 'success' | 'warning' | 'info'" },
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
		animated: {
			control: 'boolean',
			description: 'Enable pulse animation',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Animation',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof PulseBadge>

export const Default: Story = {
	args: {
		text: 'New Feature',
		variant: 'primary',
		size: 'md',
		animated: true,
	},
}

export const AllColors: Story = {
	render: () => (
		<div className="flex flex-wrap gap-4">
			<PulseBadge text="Primary" variant="primary" />
			<PulseBadge text="Success" variant="success" />
			<PulseBadge text="Warning" variant="warning" />
			<PulseBadge text="Info" variant="info" />
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available color variants.',
			},
		},
	},
}

export const WithIcon: Story = {
	render: () => (
		<div className="space-y-6">
			<div>
				<h3 className="text-sm font-semibold mb-3">Sizes</h3>
				<div className="flex flex-wrap items-center gap-4">
					<PulseBadge text="Small" size="sm" variant="primary" />
					<PulseBadge text="Medium" size="md" variant="primary" />
					<PulseBadge text="Large" size="lg" variant="primary" />
				</div>
			</div>
			<div>
				<h3 className="text-sm font-semibold mb-3">Animation States</h3>
				<div className="flex flex-wrap gap-4">
					<PulseBadge text="Animated" animated variant="success" />
					<PulseBadge text="Static" animated={false} variant="success" />
				</div>
			</div>
			<div>
				<h3 className="text-sm font-semibold mb-3">Use Cases</h3>
				<div className="flex flex-wrap gap-4">
					<PulseBadge text="🔴 Live" variant="primary" size="sm" />
					<PulseBadge text="✨ New" variant="info" size="sm" />
					<PulseBadge text="⚡ Beta" variant="warning" size="sm" />
					<PulseBadge text="✅ Active" variant="success" size="sm" />
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All sizes, animation states, and common use cases.',
			},
		},
	},
}

export const InHeroContext: Story = {
	render: () => (
		<div className="w-full max-w-3xl text-center">
			<div className="mb-4 flex justify-center">
				<PulseBadge text="🚀 Now in Public Beta" variant="primary" size="md" />
			</div>
			<h1 className="text-4xl md:text-5xl font-bold mb-4">Private LLM Hosting in the Netherlands</h1>
			<p className="text-lg text-muted-foreground mb-8">
				GDPR-compliant, self-hosted AI infrastructure for enterprises and developers
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
	),
	parameters: {
		docs: {
			description: {
				story: 'PulseBadge as used in HeroSection, announcing new features or product status.',
			},
		},
	},
}
