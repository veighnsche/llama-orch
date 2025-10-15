import type { Meta, StoryObj } from '@storybook/react'
import { Zap, Shield, Check, AlertCircle } from 'lucide-react'
import { IconPlate } from './IconPlate'

const meta: Meta<typeof IconPlate> = {
	title: 'Molecules/IconPlate',
	component: IconPlate,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
IconPlate is a reusable icon container molecule that consolidates 15+ instances of icon wrapper patterns across the commercial site. It provides consistent styling for icons in stats, features, cards, and list items.

## Composition
This molecule is composed of:
- **Container div**: Provides background, padding, and shape
- **Icon slot**: Accepts any ReactNode (typically Lucide icons)
- **Color tones**: Primary, muted, success, warning

## When to Use
- Stats grids (icon + metric)
- Feature lists (icon + description)
- Card headers (icon + title)
- List items (icon + text)
- Status indicators (icon + state)

## Variants
- **Sizes**: sm, md, lg
- **Tones**: primary, muted, success, warning
- **Shapes**: square (rounded), circle

## Used In Commercial Site
Used in 3+ organisms including:
- StatsGrid (stat icons)
- FeatureCard (feature icons)
- BulletListItem (checkmark icons)
- BenefitCallout (benefit icons)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			control: false,
			description: 'Icon element to display',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
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
		tone: {
			control: 'select',
			options: ['primary', 'muted', 'success', 'warning', 'chart-1', 'chart-2', 'chart-3', 'chart-4', 'chart-5'],
			description: 'Color tone',
			table: {
				type: { summary: "'primary' | 'muted' | 'success' | 'warning' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'" },
				defaultValue: { summary: 'primary' },
				category: 'Appearance',
			},
		},
		shape: {
			control: 'select',
			options: ['square', 'rounded', 'circle'],
			description: 'Shape variant',
			table: {
				type: { summary: "'square' | 'rounded' | 'circle'" },
				defaultValue: { summary: 'square' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof IconPlate>

export const Default: Story = {
	args: {
		icon: Zap,
		size: 'md',
		tone: 'primary',
		shape: 'square',
	},
}

export const AllVariants: Story = {
	render: () => (
		<div className="space-y-6">
			<div>
				<h3 className="text-sm font-semibold mb-3">Tones</h3>
				<div className="flex items-center gap-4">
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={Zap} tone="primary" />
						<span className="text-xs text-muted-foreground">Primary</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={Shield} tone="muted" />
						<span className="text-xs text-muted-foreground">Muted</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={Check} tone="success" />
						<span className="text-xs text-muted-foreground">Success</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={AlertCircle} tone="warning" />
						<span className="text-xs text-muted-foreground">Warning</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={Zap} tone="chart-1" />
						<span className="text-xs text-muted-foreground">Chart-1</span>
					</div>
				</div>
			</div>
			<div>
				<h3 className="text-sm font-semibold mb-3">Shapes</h3>
				<div className="flex items-center gap-4">
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={Zap} shape="square" />
						<span className="text-xs text-muted-foreground">Square</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={Zap} shape="rounded" />
						<span className="text-xs text-muted-foreground">Rounded</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<IconPlate icon={Zap} shape="circle" />
						<span className="text-xs text-muted-foreground">Circle</span>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available tone and shape variants.',
			},
		},
	},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex items-end gap-6">
			<div className="flex flex-col items-center gap-2">
				<IconPlate icon={Zap} size="sm" />
				<span className="text-xs text-muted-foreground">Small</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<IconPlate icon={Zap} size="md" />
				<span className="text-xs text-muted-foreground">Medium</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<IconPlate icon={Zap} size="lg" />
				<span className="text-xs text-muted-foreground">Large</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<IconPlate icon={Zap} size="xl" />
				<span className="text-xs text-muted-foreground">X-Large</span>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available size variants.',
			},
		},
	},
}

export const InUseCaseContext: Story = {
	render: () => (
		<div className="w-full max-w-2xl">
			<div className="mb-6">
				<h2 className="text-2xl font-bold mb-2">Use Cases</h2>
				<p className="text-muted-foreground">Common scenarios for private LLM hosting</p>
			</div>
			<div className="space-y-4">
				<div className="flex items-start gap-4 p-4 rounded-lg border border-border bg-card">
					<IconPlate icon={Shield} tone="primary" size="md" />
					<div>
						<h3 className="font-semibold mb-1">Healthcare Data Processing</h3>
						<p className="text-sm text-muted-foreground">
							Process sensitive patient data with GDPR-compliant infrastructure
						</p>
					</div>
				</div>
				<div className="flex items-start gap-4 p-4 rounded-lg border border-border bg-card">
					<IconPlate icon={Check} tone="success" size="md" />
					<div>
						<h3 className="font-semibold mb-1">Legal Document Analysis</h3>
						<p className="text-sm text-muted-foreground">
							Analyze contracts and legal documents with complete data sovereignty
						</p>
					</div>
				</div>
				<div className="flex items-start gap-4 p-4 rounded-lg border border-border bg-card">
					<IconPlate icon={Zap} tone="primary" size="md" />
					<div>
						<h3 className="font-semibold mb-1">Real-Time Customer Support</h3>
						<p className="text-sm text-muted-foreground">
							Power chatbots with sub-50ms latency for instant responses
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'IconPlate as used in use case lists, showing icons with descriptions.',
			},
		},
	},
}
