import type { Meta, StoryObj } from '@storybook/react'
import { CheckListItem } from './CheckListItem'

const meta: Meta<typeof CheckListItem> = {
	title: 'Molecules/Pricing/CheckListItem',
	component: CheckListItem,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The CheckListItem molecule displays a list item with a check icon, used for feature lists, benefits, and included items.

## Composition
This molecule is composed of:
- **Check Icon**: Lucide Check icon
- **Text**: Item content

## When to Use
- In pricing plan feature lists
- In benefit descriptions
- In "what's included" sections
- In comparison tables
- Anywhere you need to show included/completed items

## Variants
- **Color Variants**: success (green), primary (brand), muted (gray)
- **Size Variants**: sm, md, lg

## Used In Commercial Site
- **Pricing Page**: Plan feature lists
- **Features Page**: Feature benefits
- **Enterprise Page**: Included services
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		text: {
			control: 'text',
			description: 'List item text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		variant: {
			control: 'select',
			options: ['success', 'primary', 'muted'],
			description: 'Color variant',
			table: {
				type: { summary: "'success' | 'primary' | 'muted'" },
				defaultValue: { summary: 'success' },
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
	},
}

export default meta
type Story = StoryObj<typeof CheckListItem>

export const Default: Story = {
	args: {
		text: 'OpenAI-compatible API',
		variant: 'success',
		size: 'md',
	},
}

export const Success: Story = {
	args: {
		text: 'Full GDPR compliance',
		variant: 'success',
		size: 'md',
	},
	parameters: {
		docs: {
			description: {
				story: 'Success variant with green check icon, used for positive features and benefits.',
			},
		},
	},
}

export const Primary: Story = {
	args: {
		text: 'Multi-GPU support',
		variant: 'primary',
		size: 'md',
	},
	parameters: {
		docs: {
			description: {
				story: 'Primary variant with brand-colored check icon.',
			},
		},
	},
}

export const Muted: Story = {
	args: {
		text: 'Basic monitoring',
		variant: 'muted',
		size: 'md',
	},
	parameters: {
		docs: {
			description: {
				story: 'Muted variant with gray check icon, used for standard features.',
			},
		},
	},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex flex-col gap-8 p-8 max-w-2xl">
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Small</h3>
				<ul className="space-y-2">
					<CheckListItem text="Small check list item" size="sm" variant="success" />
					<CheckListItem text="Used in compact layouts" size="sm" variant="success" />
				</ul>
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Medium (Default)</h3>
				<ul className="space-y-2">
					<CheckListItem text="Medium check list item" size="md" variant="success" />
					<CheckListItem text="Default size for most use cases" size="md" variant="success" />
				</ul>
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Large</h3>
				<ul className="space-y-2">
					<CheckListItem text="Large check list item" size="lg" variant="success" />
					<CheckListItem text="Used in hero sections and emphasis" size="lg" variant="success" />
				</ul>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available size variants from small to large.',
			},
		},
	},
}

export const AllVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-4 p-8 max-w-2xl">
			<h3 className="text-lg font-semibold text-foreground">Color Variants</h3>
			<ul className="space-y-3">
				<CheckListItem text="Success variant (green)" variant="success" size="md" />
				<CheckListItem text="Primary variant (brand color)" variant="primary" size="md" />
				<CheckListItem text="Muted variant (gray)" variant="muted" size="md" />
			</ul>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All color variants: success for positive features, primary for brand emphasis, muted for standard items.',
			},
		},
	},
}

export const PricingContext: Story = {
	render: () => (
		<div className="w-full max-w-md p-8">
			<div className="rounded-lg border border-border bg-card p-6">
				<h3 className="mb-4 text-xl font-bold text-foreground">Pro Plan</h3>
				<p className="mb-6 text-3xl font-bold text-foreground">
					€99<span className="text-base font-normal text-muted-foreground">/month</span>
				</p>
				<ul className="space-y-3">
					<CheckListItem text="Up to 10 GPU workers" variant="success" size="md" />
					<CheckListItem text="OpenAI-compatible API" variant="success" size="md" />
					<CheckListItem text="Multi-GPU support" variant="success" size="md" />
					<CheckListItem text="Priority support" variant="success" size="md" />
					<CheckListItem text="Custom deployment" variant="success" size="md" />
				</ul>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'CheckListItem as used in pricing plan cards to show included features.',
			},
		},
	},
}
