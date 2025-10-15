import type { Meta, StoryObj } from '@storybook/react'
import { PricingTier } from './PricingTier'

const meta: Meta<typeof PricingTier> = {
	title: 'Molecules/Pricing/PricingTier',
	component: PricingTier,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The PricingTier molecule displays a pricing plan with title, price, features list, CTA button, and optional badge/footnote. Supports monthly/yearly toggle.

## Composition
This molecule is composed of:
- **Badge**: Optional badge (e.g., "Most Popular")
- **Title**: Plan name
- **Price**: Price display with period
- **Features**: List with checkmarks
- **CTA Button**: Action button
- **Footnote**: Optional fine print

## When to Use
- Pricing pages
- Plan comparison tables
- Subscription tiers
- Feature comparison

## Used In
- **PricingSection**: Displays pricing tiers (Starter, Professional, Enterprise)
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		title: {
			control: 'text',
			description: 'Plan title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		price: {
			control: 'text',
			description: 'Monthly price',
			table: {
				type: { summary: 'string | number' },
				category: 'Content',
			},
		},
		priceYearly: {
			control: 'text',
			description: 'Yearly price',
			table: {
				type: { summary: 'string | number' },
				category: 'Content',
			},
		},
		period: {
			control: 'text',
			description: 'Price period (e.g., "/month")',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		features: {
			control: 'object',
			description: 'List of features',
			table: {
				type: { summary: 'string[]' },
				category: 'Content',
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
		ctaHref: {
			control: 'text',
			description: 'Optional CTA link',
			table: {
				type: { summary: 'string' },
				category: 'Behavior',
			},
		},
		ctaVariant: {
			control: 'select',
			options: ['default', 'outline'],
			description: 'CTA button variant',
			table: {
				type: { summary: "'default' | 'outline'" },
				defaultValue: { summary: 'default' },
				category: 'Appearance',
			},
		},
		highlighted: {
			control: 'boolean',
			description: 'Highlight this tier',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Appearance',
			},
		},
		badge: {
			control: 'text',
			description: 'Optional badge text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		footnote: {
			control: 'text',
			description: 'Optional footnote',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		isYearly: {
			control: 'boolean',
			description: 'Show yearly pricing',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Behavior',
			},
		},
		saveBadge: {
			control: 'text',
			description: 'Savings badge for yearly',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof PricingTier>

export const Default: Story = {
	args: {
		title: 'Professional',
		price: '€299',
		period: '/month',
		features: [
			'Up to 10 workers',
			'100k requests/month',
			'Email support',
			'99.9% uptime SLA',
		],
		ctaText: 'Start Free Trial',
		ctaHref: '/signup',
	},
}

export const Highlighted: Story = {
	args: {
		title: 'Professional',
		price: '€299',
		priceYearly: '€2,990',
		period: '/month',
		features: [
			'Up to 10 workers',
			'100k requests/month',
			'Priority email support',
			'99.9% uptime SLA',
			'Advanced monitoring',
		],
		ctaText: 'Start Free Trial',
		ctaHref: '/signup',
		highlighted: true,
		badge: 'Most Popular',
		footnote: 'No credit card required',
		saveBadge: 'Save 17%',
	},
}

export const WithFeatures: Story = {
	args: {
		title: 'Enterprise',
		price: 'Custom',
		features: [
			'Unlimited workers',
			'Unlimited requests',
			'Dedicated support',
			'99.99% uptime SLA',
			'Custom integrations',
			'On-premise deployment',
			'Training & onboarding',
		],
		ctaText: 'Contact Sales',
		ctaHref: '/contact',
		ctaVariant: 'outline',
		footnote: 'Custom pricing based on your needs',
	},
}

export const InPricingContext: Story = {
	render: () => (
		<div className="w-full max-w-6xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: PricingTier in PricingSection organism
			</div>
			<div className="grid gap-6 md:grid-cols-3">
				<PricingTier
					title="Starter"
					price="€99"
					period="/month"
					features={[
						'Up to 3 workers',
						'10k requests/month',
						'Community support',
						'99% uptime SLA',
					]}
					ctaText="Start Free Trial"
					ctaHref="/signup"
					ctaVariant="outline"
					footnote="Perfect for small teams"
				/>
				<PricingTier
					title="Professional"
					price="€299"
					priceYearly="€2,990"
					period="/month"
					features={[
						'Up to 10 workers',
						'100k requests/month',
						'Priority email support',
						'99.9% uptime SLA',
						'Advanced monitoring',
					]}
					ctaText="Start Free Trial"
					ctaHref="/signup"
					highlighted={true}
					badge="Most Popular"
					footnote="No credit card required"
					saveBadge="Save 17%"
				/>
				<PricingTier
					title="Enterprise"
					price="Custom"
					features={[
						'Unlimited workers',
						'Unlimited requests',
						'Dedicated support',
						'99.99% uptime SLA',
						'Custom integrations',
					]}
					ctaText="Contact Sales"
					ctaHref="/contact"
					ctaVariant="outline"
					footnote="Custom pricing"
				/>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'PricingTier as used in the PricingSection organism, showing three pricing tiers with the middle one highlighted.',
			},
		},
	},
}
