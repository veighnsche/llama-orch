import type { Meta, StoryObj } from '@storybook/react'
import { Zap, Shield, Cpu, Database, Cloud, Lock } from 'lucide-react'
import { IconBox } from './IconBox'

const meta: Meta<typeof IconBox> = {
	title: 'Molecules/UI/IconBox',
	component: IconBox,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
IconBox is a versatile icon container molecule that provides consistent styling for icons across the commercial site. It supports multiple sizes, colors, and shape variants.

## Composition
This molecule is composed of:
- **Container div**: Provides background, padding, and shape
- **Lucide icon**: Any Lucide icon component
- **Color system**: Uses design tokens for consistent theming

## When to Use
- Feature cards (icon + title + description)
- Benefit lists (icon + text)
- Step indicators (numbered or icon-based)
- Service cards (icon representing service type)
- Anywhere you need a styled icon container

## Variants
- **Sizes**: sm, md, lg, xl
- **Shapes**: rounded (default), circle, square
- **Colors**: primary, chart-1 through chart-5

## Used In Commercial Site
Used in 5+ organisms including:
- FeaturesSection (feature icons)
- BenefitsSection (benefit icons)
- HowItWorksSection (step icons)
- ServicesSection (service icons)
- ComparisonSection (comparison icons)
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
			options: ['primary', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
			description: 'Icon color (Tailwind class)',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'primary' },
				category: 'Appearance',
			},
		},
		size: {
			control: 'select',
			options: ['sm', 'md', 'lg', 'xl'],
			description: 'Size variant',
			table: {
				type: { summary: "'sm' | 'md' | 'lg' | 'xl'" },
				defaultValue: { summary: 'md' },
				category: 'Appearance',
			},
		},
		variant: {
			control: 'select',
			options: ['rounded', 'circle', 'square'],
			description: 'Shape variant',
			table: {
				type: { summary: "'rounded' | 'circle' | 'square'" },
				defaultValue: { summary: 'rounded' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof IconBox>

export const Default: Story = {
	args: {
		icon: Zap,
		color: 'primary',
		size: 'md',
		variant: 'rounded',
	},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex items-end gap-6">
			<div className="flex flex-col items-center gap-2">
				<IconBox icon={Zap} size="sm" />
				<span className="text-xs text-muted-foreground">Small</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<IconBox icon={Shield} size="md" />
				<span className="text-xs text-muted-foreground">Medium</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<IconBox icon={Cpu} size="lg" />
				<span className="text-xs text-muted-foreground">Large</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<IconBox icon={Database} size="xl" />
				<span className="text-xs text-muted-foreground">Extra Large</span>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available size variants from small to extra large.',
			},
		},
	},
}

export const WithLink: Story = {
	render: () => (
		<div className="grid grid-cols-3 gap-6">
			<div className="flex flex-col items-center gap-3 p-4 rounded-lg border border-border hover:border-primary/40 transition-colors cursor-pointer">
				<IconBox icon={Cloud} size="lg" color="chart-2" />
				<h3 className="font-semibold">Cloud Hosting</h3>
				<p className="text-sm text-muted-foreground text-center">Deploy to the cloud</p>
			</div>
			<div className="flex flex-col items-center gap-3 p-4 rounded-lg border border-border hover:border-primary/40 transition-colors cursor-pointer">
				<IconBox icon={Lock} size="lg" color="chart-3" />
				<h3 className="font-semibold">Security</h3>
				<p className="text-sm text-muted-foreground text-center">Enterprise-grade security</p>
			</div>
			<div className="flex flex-col items-center gap-3 p-4 rounded-lg border border-border hover:border-primary/40 transition-colors cursor-pointer">
				<IconBox icon={Zap} size="lg" color="chart-4" />
				<h3 className="font-semibold">Performance</h3>
				<p className="text-sm text-muted-foreground text-center">Lightning-fast inference</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'IconBox used in clickable feature cards with hover effects.',
			},
		},
	},
}

export const InFeaturesContext: Story = {
	render: () => (
		<div className="w-full max-w-4xl">
			<div className="mb-8 text-center">
				<h2 className="text-3xl font-bold mb-2">Key Features</h2>
				<p className="text-muted-foreground">Everything you need for private LLM hosting</p>
			</div>
			<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
				<div className="p-6 bg-card rounded-lg border border-border">
					<IconBox icon={Shield} size="lg" color="primary" className="mb-4" />
					<h3 className="font-semibold mb-2">GDPR Compliant</h3>
					<p className="text-sm text-muted-foreground">100% compliant with Dutch and EU data protection laws</p>
				</div>
				<div className="p-6 bg-card rounded-lg border border-border">
					<IconBox icon={Zap} size="lg" color="chart-1" className="mb-4" />
					<h3 className="font-semibold mb-2">Fast Inference</h3>
					<p className="text-sm text-muted-foreground">Sub-50ms latency for real-time applications</p>
				</div>
				<div className="p-6 bg-card rounded-lg border border-border">
					<IconBox icon={Cpu} size="lg" color="chart-2" className="mb-4" />
					<h3 className="font-semibold mb-2">GPU Acceleration</h3>
					<p className="text-sm text-muted-foreground">CUDA and Metal support for maximum performance</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'IconBox as used in FeaturesSection organism, showing feature icons with descriptions.',
			},
		},
	},
}
