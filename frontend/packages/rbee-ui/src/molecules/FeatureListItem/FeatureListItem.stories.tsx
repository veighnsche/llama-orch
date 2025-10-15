import type { Meta, StoryObj } from '@storybook/react'
import { FeatureListItem } from './FeatureListItem'
import { Zap, Shield, Cpu, Rocket, Lock, Globe } from 'lucide-react'

const meta: Meta<typeof FeatureListItem> = {
	title: 'Molecules/FeatureListItem',
	component: FeatureListItem,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The FeatureListItem molecule displays a feature with an icon, bold title, and description text.

## Composition
This molecule is composed of:
- **IconBox**: Visual icon indicator
- **Title**: Bold feature name
- **Description**: Explanatory text

## When to Use
- In "What is X?" sections
- In feature comparison lists
- In value proposition sections
- In product overview pages
- Anywhere you need icon + title + description lists

## Variants
- **Icon Colors**: primary, chart-1, chart-2, chart-3, chart-4, chart-5
- **Icon Variants**: rounded, square
- **Icon Sizes**: sm, md, lg

## Difference from BulletListItem
- **FeatureListItem**: Icon-based, two-line (title + description), for feature explanations
- **BulletListItem**: Simple bullet (dot/check/arrow), single-line or with description, for quick lists

## Used In Commercial Site
- **What is rbee Section**: Core value propositions with icons
- **Features Page**: Detailed feature lists
- **About Page**: Company values and principles
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
		title: {
			control: 'text',
			description: 'Feature title (bold part)',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		description: {
			control: 'text',
			description: 'Feature description',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		iconColor: {
			control: 'select',
			options: ['primary', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
			description: 'Icon color variant',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'primary' },
				category: 'Appearance',
			},
		},
		iconVariant: {
			control: 'select',
			options: ['rounded', 'square'],
			description: 'Icon container shape',
			table: {
				type: { summary: "'rounded' | 'square'" },
				defaultValue: { summary: 'rounded' },
				category: 'Appearance',
			},
		},
		iconSize: {
			control: 'select',
			options: ['sm', 'md', 'lg'],
			description: 'Icon size',
			table: {
				type: { summary: "'sm' | 'md' | 'lg'" },
				defaultValue: { summary: 'sm' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof FeatureListItem>

export const Default: Story = {
	args: {
		icon: Zap,
		title: 'Independence',
		description: 'Build on your hardware. No surprise model or pricing changes.',
		iconColor: 'primary',
		iconVariant: 'rounded',
		iconSize: 'sm',
	},
}

export const WithShield: Story = {
	args: {
		icon: Shield,
		title: 'Privacy',
		description: 'Code and data never leave your network.',
		iconColor: 'primary',
		iconVariant: 'rounded',
		iconSize: 'sm',
	},
}

export const WithCpu: Story = {
	args: {
		icon: Cpu,
		title: 'All GPUs together',
		description: 'CUDA, Metal, and CPU—scheduled as one.',
		iconColor: 'primary',
		iconVariant: 'rounded',
		iconSize: 'sm',
	},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex flex-col gap-6 p-8 max-w-2xl">
			<h3 className="text-lg font-semibold text-foreground">Icon Sizes</h3>
			<ul className="space-y-4">
				<FeatureListItem
					icon={Rocket}
					title="Small Icon"
					description="Default size for compact lists"
					iconSize="sm"
					iconColor="primary"
				/>
				<FeatureListItem
					icon={Rocket}
					title="Medium Icon"
					description="Balanced size for standard sections"
					iconSize="md"
					iconColor="primary"
				/>
				<FeatureListItem
					icon={Rocket}
					title="Large Icon"
					description="Prominent size for hero sections"
					iconSize="lg"
					iconColor="primary"
				/>
			</ul>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Different icon sizes: sm (default), md, and lg.',
			},
		},
	},
}

export const AllColors: Story = {
	render: () => (
		<div className="flex flex-col gap-6 p-8 max-w-2xl">
			<h3 className="text-lg font-semibold text-foreground">Icon Colors</h3>
			<ul className="space-y-4">
				<FeatureListItem
					icon={Lock}
					title="Primary Color"
					description="Default brand color for key features"
					iconColor="primary"
				/>
				<FeatureListItem
					icon={Lock}
					title="Chart-1 Color"
					description="Alternative color for variety"
					iconColor="chart-1"
				/>
				<FeatureListItem
					icon={Lock}
					title="Chart-2 Color"
					description="Another color option"
					iconColor="chart-2"
				/>
				<FeatureListItem
					icon={Lock}
					title="Chart-3 Color"
					description="Success/positive color"
					iconColor="chart-3"
				/>
				<FeatureListItem
					icon={Lock}
					title="Chart-4 Color"
					description="Warning/attention color"
					iconColor="chart-4"
				/>
				<FeatureListItem
					icon={Lock}
					title="Chart-5 Color"
					description="Info/neutral color"
					iconColor="chart-5"
				/>
			</ul>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available icon color variants.',
			},
		},
	},
}

export const IconVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-6 p-8 max-w-2xl">
			<h3 className="text-lg font-semibold text-foreground">Icon Variants</h3>
			<ul className="space-y-4">
				<FeatureListItem
					icon={Globe}
					title="Rounded Variant"
					description="Softer, friendlier appearance with rounded corners"
					iconVariant="rounded"
					iconColor="primary"
				/>
				<FeatureListItem
					icon={Globe}
					title="Square Variant"
					description="Sharp, technical appearance with square corners"
					iconVariant="square"
					iconColor="primary"
				/>
			</ul>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Icon container shape variants: rounded (default) and square.',
			},
		},
	},
}

export const RealWorldExample: Story = {
	render: () => (
		<div className="flex flex-col gap-6 p-8 max-w-2xl bg-secondary rounded-lg">
			<h3 className="text-2xl font-semibold text-foreground">What is rbee?</h3>
			<p className="text-muted-foreground">
				rbee is an open-source AI orchestration platform that unifies every computer in your home or office into a
				single, OpenAI-compatible AI cluster.
			</p>
			<ul className="space-y-3 text-base text-foreground">
				<FeatureListItem
					icon={Zap}
					title="Independence"
					description="Build on your hardware. No surprise model or pricing changes."
					iconColor="primary"
					iconVariant="rounded"
					iconSize="sm"
				/>
				<FeatureListItem
					icon={Shield}
					title="Privacy"
					description="Code and data never leave your network."
					iconColor="primary"
					iconVariant="rounded"
					iconSize="sm"
				/>
				<FeatureListItem
					icon={Cpu}
					title="All GPUs together"
					description="CUDA, Metal, and CPU—scheduled as one."
					iconColor="primary"
					iconVariant="rounded"
					iconSize="sm"
				/>
			</ul>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Real-world usage example from the "What is rbee?" section.',
			},
		},
	},
}
