import type { Meta, StoryObj } from '@storybook/react'
import { BulletListItem } from './BulletListItem'

const meta: Meta<typeof BulletListItem> = {
	title: 'Molecules/Content/BulletListItem',
	component: BulletListItem,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The BulletListItem molecule displays a list item with customizable bullet style, color, and optional description and meta text.

## Composition
This molecule is composed of:
- **Bullet**: Dot, check, or arrow indicator
- **Title**: Main text content
- **Description** (optional): Secondary text below title
- **Meta** (optional): Right-aligned metadata

## When to Use
- In feature lists
- In benefit descriptions
- In step-by-step instructions
- In comparison tables
- Anywhere you need styled list items

## Variants
- **Bullet Variants**: dot, check, arrow
- **Color Variants**: primary, chart-1, chart-2, chart-3, chart-4, chart-5
- **With/Without Description**: Optional secondary text
- **With/Without Meta**: Optional right-aligned text

## Used In Commercial Site
- **Features Page**: Feature lists with check bullets
- **Pricing Page**: Plan features with check bullets
- **Developers Page**: API features with arrow bullets
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		title: {
			control: 'text',
			description: 'Item title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		description: {
			control: 'text',
			description: 'Optional description text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		meta: {
			control: 'text',
			description: 'Optional right-aligned meta text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		color: {
			control: 'select',
			options: ['primary', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
			description: 'Bullet color',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'chart-3' },
				category: 'Appearance',
			},
		},
		variant: {
			control: 'select',
			options: ['dot', 'check', 'arrow'],
			description: 'Bullet style',
			table: {
				type: { summary: "'dot' | 'check' | 'arrow'" },
				defaultValue: { summary: 'dot' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof BulletListItem>

export const Default: Story = {
	args: {
		title: 'OpenAI-compatible API',
		variant: 'dot',
		color: 'chart-3',
	},
}

export const WithDescription: Story = {
	args: {
		title: 'Multi-GPU Support',
		description: 'Automatically distribute workloads across multiple GPUs for maximum throughput',
		variant: 'check',
		color: 'chart-3',
	},
}

export const WithMeta: Story = {
	args: {
		title: 'Docker Deployment',
		meta: '5 min',
		variant: 'arrow',
		color: 'primary',
	},
}

export const AllVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-8 p-8 max-w-2xl">
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Dot Variant</h3>
				<ul className="space-y-3">
					<BulletListItem title="Simple dot bullet" variant="dot" color="chart-3" />
					<BulletListItem
						title="Dot with description"
						description="Additional context below the title"
						variant="dot"
						color="chart-3"
					/>
				</ul>
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Check Variant</h3>
				<ul className="space-y-3">
					<BulletListItem title="Check bullet for completed items" variant="check" color="chart-3" />
					<BulletListItem
						title="Check with description"
						description="Used for feature lists and benefits"
						variant="check"
						color="chart-3"
					/>
				</ul>
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Arrow Variant</h3>
				<ul className="space-y-3">
					<BulletListItem title="Arrow bullet for action items" variant="arrow" color="primary" />
					<BulletListItem
						title="Arrow with meta"
						meta="2 min"
						variant="arrow"
						color="primary"
					/>
				</ul>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All bullet variants: dot (neutral), check (completed/included), arrow (action/navigation).',
			},
		},
	},
}

export const AllColors: Story = {
	render: () => (
		<div className="flex flex-col gap-4 p-8 max-w-2xl">
			<h3 className="text-lg font-semibold text-foreground">Color Variants</h3>
			<ul className="space-y-3">
				<BulletListItem title="Primary color" variant="check" color="primary" />
				<BulletListItem title="Chart-1 color" variant="check" color="chart-1" />
				<BulletListItem title="Chart-2 color" variant="check" color="chart-2" />
				<BulletListItem title="Chart-3 color (default)" variant="check" color="chart-3" />
				<BulletListItem title="Chart-4 color" variant="check" color="chart-4" />
				<BulletListItem title="Chart-5 color" variant="check" color="chart-5" />
			</ul>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available color variants for the bullet indicator.',
			},
		},
	},
}
