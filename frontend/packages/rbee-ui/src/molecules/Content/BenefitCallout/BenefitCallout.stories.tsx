import type { Meta, StoryObj } from '@storybook/react'
import { Check, Zap, Shield, Info } from 'lucide-react'
import { BenefitCallout } from './BenefitCallout'

const meta: Meta<typeof BenefitCallout> = {
	title: 'Molecules/Content/BenefitCallout',
	component: BenefitCallout,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The BenefitCallout molecule displays a highlighted benefit or key point with optional icon and color variant. Used for emphasis and visual hierarchy.

## Composition
This molecule is composed of:
- **Icon**: Optional icon (defaults to checkmark)
- **Text**: Benefit text
- **Variant**: Color scheme (success, primary, info, warning)

## When to Use
- Highlighting key benefits
- Success messages
- Important callouts
- Feature emphasis
- Value propositions

## Used In
- **FeaturesSection**: Displays key benefits and value propositions
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		text: {
			control: 'text',
			description: 'Callout text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		variant: {
			control: 'select',
			options: ['success', 'primary', 'info', 'warning'],
			description: 'Color variant',
			table: {
				type: { summary: "'success' | 'primary' | 'info' | 'warning'" },
				defaultValue: { summary: 'success' },
				category: 'Appearance',
			},
		},
		icon: {
			control: false,
			description: 'Optional icon (defaults to checkmark)',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof BenefitCallout>

export const Default: Story = {
	args: {
		text: 'Your data never leaves your infrastructure',
		variant: 'success',
	},
}

export const WithIcon: Story = {
	args: {
		text: 'Lightning-fast inference with GPU acceleration',
		variant: 'primary',
		icon: <Zap className="h-4 w-4" />,
	},
}

export const AllVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-4 p-8 max-w-2xl">
			<h3 className="text-lg font-semibold text-foreground mb-2">All Variants</h3>
			<BenefitCallout
				text="Your data never leaves your infrastructure"
				variant="success"
			/>
			<BenefitCallout
				text="Enterprise-grade performance and reliability"
				variant="primary"
				icon={<Zap className="h-4 w-4" />}
			/>
			<BenefitCallout
				text="GDPR compliant by design"
				variant="info"
				icon={<Info className="h-4 w-4" />}
			/>
			<BenefitCallout
				text="Bank-grade security with zero-trust architecture"
				variant="success"
				icon={<Shield className="h-4 w-4" />}
			/>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available color variants with different icons.',
			},
		},
	},
}

export const InFeaturesContext: Story = {
	render: () => (
		<div className="w-full max-w-4xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: BenefitCallout in FeaturesSection organism
			</div>
			<div className="rounded-2xl border border-border bg-card p-8 space-y-6">
				<div>
					<h2 className="text-3xl font-bold text-foreground mb-2">Why Choose rbee?</h2>
					<p className="text-muted-foreground mb-6">
						Private LLM hosting that doesn't compromise on performance or security.
					</p>
				</div>
				<div className="space-y-4">
					<BenefitCallout
						text="Your data never leaves your infrastructure"
						variant="success"
						icon={<Check className="h-4 w-4" />}
					/>
					<BenefitCallout
						text="10x faster than cloud APIs with GPU acceleration"
						variant="primary"
						icon={<Zap className="h-4 w-4" />}
					/>
					<BenefitCallout
						text="GDPR compliant with Dutch hosting"
						variant="info"
						icon={<Shield className="h-4 w-4" />}
					/>
					<BenefitCallout
						text="99.99% uptime SLA with automatic failover"
						variant="success"
						icon={<Check className="h-4 w-4" />}
					/>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'BenefitCallout as used in the FeaturesSection organism, highlighting key value propositions.',
			},
		},
	},
}
