import type { Meta, StoryObj } from '@storybook/react'
import { Building2, Code, Rocket } from 'lucide-react'
import { AudienceCard } from './AudienceCard'

const meta: Meta<typeof AudienceCard> = {
	title: 'Molecules/UI/AudienceCard',
	component: AudienceCard,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The AudienceCard molecule displays a target audience segment with icon, description, features, and CTA. Used to help visitors identify which product offering matches their needs.

## Composition
This molecule is composed of:
- **Icon**: Lucide icon representing the audience
- **Category**: Uppercase label for the audience type
- **Title**: Main heading
- **Description**: Explanatory text
- **Features List**: Bullet points with arrow indicators
- **Button**: CTA with arrow icon
- **Optional Badge**: Additional visual indicator
- **Optional Image**: Secondary visual element

## When to Use
- On the home page to segment audiences
- On landing pages to guide user journeys
- In product selection flows
- In marketing funnels

## Variants
- **Color Variants**: primary, chart-1, chart-2, chart-3, chart-4, chart-5
- **With/Without Image**: Optional secondary visual
- **With/Without Badge**: Optional badge slot
- **With/Without Decision Label**: Optional label above card

## Used In Commercial Site
- **Home Page**: Three audience cards (Developers, Enterprises, Providers)
- **Features Page**: Audience segmentation
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			description: 'Lucide icon component',
			table: {
				type: { summary: 'LucideIcon' },
				category: 'Content',
			},
		},
		category: {
			control: 'text',
			description: 'Uppercase category label',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		title: {
			control: 'text',
			description: 'Card title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		description: {
			control: 'text',
			description: 'Card description',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		features: {
			description: 'List of features',
			table: {
				type: { summary: 'string[]' },
				category: 'Content',
			},
		},
		href: {
			control: 'text',
			description: 'CTA link destination',
			table: {
				type: { summary: 'string' },
				category: 'Behavior',
			},
		},
		ctaText: {
			control: 'text',
			description: 'CTA button text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		color: {
			control: 'select',
			options: ['primary', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
			description: 'Color variant',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'primary' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof AudienceCard>

export const Default: Story = {
	args: {
		icon: Code,
		category: 'FOR DEVELOPERS',
		title: 'Build with Private AI',
		description: 'Self-hosted LLM infrastructure for your applications. Full control, zero vendor lock-in.',
		features: [
			'Deploy in minutes with Docker or Kubernetes',
			'OpenAI-compatible API for easy migration',
			'Multi-GPU support for high throughput',
		],
		href: '/developers',
		ctaText: 'Start Building',
		color: 'primary',
	},
}

export const Enterprise: Story = {
	args: {
		icon: Building2,
		category: 'FOR ENTERPRISES',
		title: 'Enterprise-Grade AI',
		description: 'GDPR-compliant, SOC2-ready AI infrastructure hosted in the Netherlands.',
		features: [
			'Full data sovereignty and compliance',
			'Dedicated support and SLAs',
			'Custom deployment options',
		],
		href: '/enterprise',
		ctaText: 'Contact Sales',
		color: 'chart-1',
	},
}

export const Provider: Story = {
	args: {
		icon: Rocket,
		category: 'FOR PROVIDERS',
		title: 'Monetize Your GPUs',
		description: 'Turn idle GPU capacity into revenue by joining the rbee network.',
		features: [
			'Earn passive income from unused GPUs',
			'Automated workload distribution',
			'Transparent pricing and payments',
		],
		href: '/providers',
		ctaText: 'Start Earning',
		color: 'chart-3',
	},
}

export const AllColors: Story = {
	render: () => (
		<div className="grid grid-cols-1 gap-6 p-8 md:grid-cols-2 lg:grid-cols-3">
			<AudienceCard
				icon={Code}
				category="PRIMARY"
				title="Primary Color"
				description="Default primary color variant for main CTAs."
				features={['Feature one', 'Feature two', 'Feature three']}
				href="#"
				ctaText="Learn More"
				color="primary"
			/>
			<AudienceCard
				icon={Building2}
				category="CHART-1"
				title="Chart-1 Color"
				description="Alternative color for secondary audiences."
				features={['Feature one', 'Feature two', 'Feature three']}
				href="#"
				ctaText="Learn More"
				color="chart-1"
			/>
			<AudienceCard
				icon={Rocket}
				category="CHART-3"
				title="Chart-3 Color"
				description="Success color for positive outcomes."
				features={['Feature one', 'Feature two', 'Feature three']}
				href="#"
				ctaText="Learn More"
				color="chart-3"
			/>
		</div>
	),
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				story: 'All available color variants. Each color creates a different visual hierarchy and emotional response.',
			},
		},
	},
}
