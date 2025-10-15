import type { Meta, StoryObj } from '@storybook/react'
import { Zap, Shield, Globe, Clock } from 'lucide-react'
import { FeatureCard } from './FeatureCard'

const meta: Meta<typeof FeatureCard> = {
	title: 'Molecules/Content/FeatureCard',
	component: FeatureCard,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The FeatureCard molecule displays a feature with icon, title, intro, bullet points, and optional link. Supports multiple sizes and icon colors.

## Composition
This molecule is composed of:
- **Icon**: Feature icon with colored background
- **Title**: Feature name
- **Intro**: Description paragraph
- **Bullets**: List with CheckItem components
- **Link**: Optional "Learn more" link
- **Children**: Optional footer content

## When to Use
- Feature showcases
- Product capabilities
- Service descriptions
- Benefit highlights

## Used In
- **AdditionalFeaturesGrid**: Displays additional features in a grid layout
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			control: false,
			description: 'Lucide icon component or ReactNode',
			table: {
				type: { summary: 'LucideIcon | ReactNode' },
				category: 'Content',
			},
		},
		title: {
			control: 'text',
			description: 'Feature title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		intro: {
			control: 'text',
			description: 'Feature description',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		bullets: {
			control: 'object',
			description: 'List of feature points',
			table: {
				type: { summary: 'string[]' },
				category: 'Content',
			},
		},
		href: {
			control: 'text',
			description: 'Optional link URL',
			table: {
				type: { summary: 'string' },
				category: 'Behavior',
			},
		},
		iconColor: {
			control: 'select',
			options: ['primary', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
			description: 'Icon background color',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'primary' },
				category: 'Appearance',
			},
		},
		hover: {
			control: 'boolean',
			description: 'Enable hover effect',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Appearance',
			},
		},
		size: {
			control: 'select',
			options: ['sm', 'md', 'lg'],
			description: 'Card size',
			table: {
				type: { summary: "'sm' | 'md' | 'lg'" },
				defaultValue: { summary: 'md' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof FeatureCard>

export const Default: Story = {
	args: {
		icon: Zap,
		title: 'Lightning Fast',
		intro: 'Optimized inference with GPU acceleration for sub-second response times.',
		bullets: [
			'CUDA and ROCm support',
			'Batch processing',
			'Model quantization',
		],
		iconColor: 'primary',
	},
}

export const WithIcon: Story = {
	args: {
		icon: Shield,
		title: 'Enterprise Security',
		intro: 'Bank-grade security with end-to-end encryption and compliance certifications.',
		bullets: [
			'AES-256 encryption',
			'SOC 2 Type II certified',
			'GDPR compliant',
			'Zero-trust architecture',
		],
		iconColor: 'chart-3',
		hover: true,
	},
}

export const WithLink: Story = {
	args: {
		icon: Globe,
		title: 'Global Infrastructure',
		intro: 'Deploy across multiple regions with automatic failover and load balancing.',
		bullets: [
			'EU and US datacenters',
			'99.99% uptime SLA',
			'Auto-scaling',
			'CDN integration',
		],
		href: '/docs/infrastructure',
		iconColor: 'chart-2',
		hover: true,
	},
}

export const InFeaturesContext: Story = {
	render: () => (
		<div className="w-full max-w-6xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: FeatureCard in AdditionalFeaturesGrid organism
			</div>
			<div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
				<FeatureCard
					icon={Zap}
					title="Lightning Fast"
					intro="Optimized inference with GPU acceleration for sub-second response times."
					bullets={[
						'CUDA and ROCm support',
						'Batch processing',
						'Model quantization',
					]}
					iconColor="primary"
					hover={true}
				/>
				<FeatureCard
					icon={Shield}
					title="Enterprise Security"
					intro="Bank-grade security with end-to-end encryption and compliance certifications."
					bullets={[
						'AES-256 encryption',
						'SOC 2 Type II certified',
						'GDPR compliant',
					]}
					iconColor="chart-3"
					hover={true}
				/>
				<FeatureCard
					icon={Clock}
					title="24/7 Monitoring"
					intro="Real-time monitoring and alerting for your entire infrastructure."
					bullets={[
						'Prometheus metrics',
						'Custom dashboards',
						'Slack/PagerDuty integration',
					]}
					iconColor="chart-4"
					hover={true}
					href="/docs/monitoring"
				/>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'FeatureCard as used in the AdditionalFeaturesGrid organism, showing three features in a grid layout.',
			},
		},
	},
}
