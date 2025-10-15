import type { Meta, StoryObj } from '@storybook/react'
import { BrandLogo } from './BrandLogo'

const meta: Meta<typeof BrandLogo> = {
	title: 'Molecules/Branding/BrandLogo',
	component: BrandLogo,
	parameters: {
		layout: 'centered',
		backgrounds: {
			default: 'dark',
		},
		docs: {
			description: {
				component: `
## Overview
The BrandLogo component combines the BrandMark (bee icon) with the "rbee" wordmark to create the complete brand identity. It's a clickable logo that typically links to the home page.

## Composition
This molecule is composed of:
- **BrandMark**: The bee icon SVG (atom)
- **Wordmark**: The "rbee" text styled with brand typography
- **Link Wrapper**: Optional Next.js Link component for navigation

## When to Use
- In the Navigation component (top-left corner of all pages)
- In the Footer component (brand identity section)
- In loading states or splash screens
- Anywhere you need the complete brand identity

## Variants
- **Small (sm)**: Compact logo for mobile navigation or footer
- **Medium (md)**: Default size for desktop navigation
- **Large (lg)**: Prominent branding in hero sections or landing pages
- **With/Without Link**: Can be rendered as a link or static element

## Used In Commercial Site
- **Navigation**: Top-left corner, links to home page, medium size on desktop, small on mobile
- **Footer**: Brand identity section, medium size, links to home page
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		size: {
			control: 'select',
			options: ['sm', 'md', 'lg'],
			description: 'Size variant of the logo',
			table: {
				type: { summary: "'sm' | 'md' | 'lg'" },
				defaultValue: { summary: 'md' },
				category: 'Appearance',
			},
		},
		priority: {
			control: 'boolean',
			description: 'Whether to prioritize loading (for above-the-fold images)',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Performance',
			},
		},
		href: {
			control: 'text',
			description: 'URL to navigate to when clicked. If undefined, renders as non-clickable element.',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'undefined' },
				category: 'Behavior',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof BrandLogo>

export const Default: Story = {
	args: {
		size: 'md',
		href: '/',
	},
}

export const Small: Story = {
	args: {
		size: 'sm',
		href: '/',
	},
}

export const Large: Story = {
	args: {
		size: 'lg',
		href: '/',
	},
}

export const WithoutLink: Story = {
	args: {
		size: 'md',
		href: undefined,
	},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex flex-col gap-8 p-8">
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Small</h3>
				<BrandLogo size="sm" />
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Medium (Default)</h3>
				<BrandLogo size="md" />
			</div>
			<div className="flex flex-col gap-4">
				<h3 className="text-lg font-semibold text-foreground">Large</h3>
				<BrandLogo size="lg" />
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All available size variants. Small is used in mobile navigation and footer, medium in desktop navigation, and large in hero sections.',
			},
		},
	},
}

export const NavigationContext: Story = {
	render: () => (
		<div className="w-full">
			<div className="mb-4 text-sm text-muted-foreground">Example: BrandLogo in Navigation component</div>
			<div className="flex items-center justify-between rounded-lg border border-border bg-card p-4">
				<BrandLogo size="md" href="/" />
				<div className="flex gap-6">
					<a href="#" className="text-sm text-muted-foreground hover:text-foreground">Features</a>
					<a href="#" className="text-sm text-muted-foreground hover:text-foreground">Pricing</a>
					<a href="#" className="text-sm text-muted-foreground hover:text-foreground">Docs</a>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'BrandLogo as it appears in the Navigation component. Positioned top-left, links to home page, medium size on desktop.',
			},
		},
	},
}

export const FooterContext: Story = {
	render: () => (
		<div className="w-full">
			<div className="mb-4 text-sm text-muted-foreground">Example: BrandLogo in Footer component</div>
			<div className="rounded-lg border border-border bg-card p-8">
				<div className="flex flex-col gap-4">
					<BrandLogo size="md" href="/" />
					<p className="max-w-xs text-sm text-muted-foreground">
						Private LLM hosting in the Netherlands. GDPR-compliant, self-hosted AI infrastructure.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'BrandLogo as it appears in the Footer component. Includes brand description below the logo.',
			},
		},
	},
}
