import type { Meta, StoryObj } from '@storybook/react'
import { Building2, Code, Rocket, ArrowRight } from 'lucide-react'
import { AudienceCard } from './AudienceCard'
import { Button } from '@rbee/ui/atoms/Button'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type { LucideIcon } from 'lucide-react'

// Original AudienceCard component (before refactor)
interface OriginalAudienceCardProps {
	icon: LucideIcon
	category: string
	title: string
	description: string
	features: string[]
	href: string
	ctaText: string
	color: string
	className?: string
}

function OriginalAudienceCard({
	icon: Icon,
	category,
	title,
	description,
	features,
	href,
	ctaText,
	color,
	className,
}: OriginalAudienceCardProps) {
	const colorClasses = {
		primary: {
			hoverBorder: 'hover:border-primary/50',
			gradient: 'from-primary/0 via-primary/0 to-primary/0 group-hover:from-primary/5 group-hover:via-primary/10',
			iconBg: 'from-primary to-primary',
			text: 'text-primary',
			button: 'bg-primary',
		},
		'chart-1': {
			hoverBorder: 'hover:border-chart-1/50',
			gradient: 'from-chart-1/0 via-chart-1/0 to-chart-1/0 group-hover:from-chart-1/5 group-hover:via-chart-1/10',
			iconBg: 'from-chart-1 to-chart-1',
			text: 'text-chart-1',
			button: 'bg-chart-1',
		},
		'chart-3': {
			hoverBorder: 'hover:border-chart-3/50',
			gradient: 'from-chart-3/0 via-chart-3/0 to-chart-3/0 group-hover:from-chart-3/5 group-hover:via-chart-3/10',
			iconBg: 'from-chart-3 to-chart-3',
			text: 'text-chart-3',
			button: 'bg-chart-3',
		},
	}

	const colors = colorClasses[color as keyof typeof colorClasses] || colorClasses.primary
	const descriptionId = `${title.toLowerCase().replace(/\s+/g, '-')}-description`

	return (
		<div className="flex flex-col">
			<div
				className={cn(
					'group relative flex flex-col overflow-hidden border-border bg-card backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] rounded-lg border p-6',
					colors.hoverBorder,
					className,
				)}
			>
				<div
					className={cn(
						'absolute inset-0 -z-10 bg-gradient-to-br opacity-0 transition-all duration-500 group-hover:to-transparent group-hover:opacity-100',
						colors.gradient,
					)}
				/>

				<div className="mb-6 flex items-center gap-3">
					<div
						className={cn(
							'flex h-14 w-14 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg',
							colors.iconBg,
						)}
					>
						<Icon className="h-7 w-7 text-primary-foreground" aria-hidden="true" />
					</div>
				</div>

				<div className={cn('mb-2 text-sm font-medium uppercase tracking-wider', colors.text)}>{category}</div>
				<h3 className="mb-3 text-2xl font-semibold text-card-foreground">{title}</h3>
				<p id={descriptionId} className="mb-6 text-sm leading-relaxed text-muted-foreground sm:text-base">
					{description}
				</p>

				<ul className="mb-8 space-y-3 text-sm text-muted-foreground sm:text-base">
					{features.map((feature, index) => (
						<li key={index} className="flex items-start gap-2">
							<span className={cn('mt-1', colors.text)} aria-hidden="true">
								→
							</span>
							<span>{feature}</span>
						</li>
					))}
				</ul>

				<div className="flex-1" />

				<Link href={href}>
					<Button className={cn('w-full', colors.button)} aria-describedby={descriptionId}>
						{ctaText}
						<ArrowRight className="ml-2 h-4 w-4" aria-hidden="true" />
					</Button>
				</Link>
			</div>
		</div>
	)
}

const meta: Meta<typeof AudienceCard> = {
	title: 'Molecules/AudienceCard',
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

export const OriginalVsNew: Story = {
	render: () => (
		<div className="space-y-12 p-8 bg-background">
			<div>
				<h2 className="text-2xl font-bold mb-4 text-foreground">Original Design (Plain div, no Card atom)</h2>
				<div className="grid grid-cols-1 gap-6 md:grid-cols-3">
					<OriginalAudienceCard
						icon={Code}
						category="FOR DEVELOPERS"
						title="Build on Your Hardware"
						description="Power Zed, Cursor, and your own agents on YOUR GPUs. OpenAI-compatible—drop-in, zero API fees."
						features={[
							'Zero API costs, unlimited usage',
							'Your code stays on your network',
							'Agentic API + TypeScript utils',
						]}
						href="#"
						ctaText="Explore Developer Path"
						color="primary"
					/>
					<OriginalAudienceCard
						icon={Building2}
						category="FOR GPU OWNERS"
						title="Monetize Your Hardware"
						description="Join the rbee marketplace and earn from gaming rigs to server farms—set price, stay in control."
						features={[
							'Set pricing & availability',
							'Audit trails and payouts',
							'Passive income from idle GPUs',
						]}
						href="#"
						ctaText="Become a Provider"
						color="chart-1"
					/>
					<OriginalAudienceCard
						icon={Rocket}
						category="FOR ENTERPRISE"
						title="Compliance & Security"
						description="EU-native compliance, audit trails, and zero-trust architecture—from day one."
						features={[
							'GDPR with 7-year retention',
							'SOC2 & ISO 27001 aligned',
							'Private cloud or on-prem',
						]}
						href="#"
						ctaText="Enterprise Solutions"
						color="chart-3"
					/>
				</div>
			</div>

			<div>
				<h2 className="text-2xl font-bold mb-4 text-foreground">New Design (Card atom + ButtonCardFooter)</h2>
				<div className="grid grid-cols-1 gap-6 md:grid-cols-3">
					<AudienceCard
						icon={Code}
						category="FOR DEVELOPERS"
						title="Build on Your Hardware"
						description="Power Zed, Cursor, and your own agents on YOUR GPUs. OpenAI-compatible—drop-in, zero API fees."
						features={[
							'Zero API costs, unlimited usage',
							'Your code stays on your network',
							'Agentic API + TypeScript utils',
						]}
						href="#"
						ctaText="Explore Developer Path"
						color="primary"
					/>
					<AudienceCard
						icon={Building2}
						category="FOR GPU OWNERS"
						title="Monetize Your Hardware"
						description="Join the rbee marketplace and earn from gaming rigs to server farms—set price, stay in control."
						features={[
							'Set pricing & availability',
							'Audit trails and payouts',
							'Passive income from idle GPUs',
						]}
						href="#"
						ctaText="Become a Provider"
						color="chart-1"
					/>
					<AudienceCard
						icon={Rocket}
						category="FOR ENTERPRISE"
						title="Compliance & Security"
						description="EU-native compliance, audit trails, and zero-trust architecture—from day one."
						features={[
							'GDPR with 7-year retention',
							'SOC2 & ISO 27001 aligned',
							'Private cloud or on-prem',
						]}
						href="#"
						ctaText="Enterprise Solutions"
						color="chart-3"
					/>
				</div>
			</div>
		</div>
	),
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				story: 'Side-by-side comparison of the original AudienceCard design vs the new refactored version with Card atom and ButtonCardFooter.',
			},
		},
	},
}
