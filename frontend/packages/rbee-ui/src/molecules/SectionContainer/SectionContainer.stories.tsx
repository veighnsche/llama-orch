import type { Meta, StoryObj } from '@storybook/react'
import { SectionContainer } from './SectionContainer'

const meta: Meta<typeof SectionContainer> = {
	title: 'Molecules/SectionContainer',
	component: SectionContainer,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
SectionContainer is the foundational layout molecule used across all page sections. It provides consistent spacing, alignment, and header structure for content sections.

## Composition
This molecule is composed of:
- **Semantic section element**: Provides proper HTML5 structure
- **Container wrapper**: Constrains content width
- **Header block**: Title, subtitle, eyebrow, kicker, actions
- **Content area**: Children slot for section content

## When to Use
- Every major page section (heroes, features, CTAs, etc.)
- Anywhere you need consistent section layout
- When you need flexible header layouts (stack or split)
- For sections with optional backgrounds and padding

## Variants
- **Background variants**: background, secondary, card, muted, subtle
- **Alignment**: start (left) or center
- **Layout**: stack (single column) or split (two-column header)
- **Padding**: lg, xl, 2xl
- **Max width**: xl through 7xl

## Used In Commercial Site
Used in 15+ organisms including:
- HeroSection (home page)
- FeaturesSection (all feature sections)
- CTASection (call-to-action sections)
- TestimonialsSection (testimonials)
- ComparisonSection (comparison tables)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		title: {
			control: 'text',
			description: 'Section title (null to skip rendering)',
			table: {
				type: { summary: 'string | ReactNode | null' },
				category: 'Content',
			},
		},
		description: {
			control: 'text',
			description: 'Optional description text below title',
			table: {
				type: { summary: 'string | ReactNode' },
				category: 'Content',
			},
		},
		eyebrow: {
			control: 'text',
			description: 'Small badge/label above title',
			table: {
				type: { summary: 'string | ReactNode' },
				category: 'Content',
			},
		},
		kicker: {
			control: 'text',
			description: 'Short lead-in sentence between eyebrow and title',
			table: {
				type: { summary: 'string | ReactNode' },
				category: 'Content',
			},
		},
		bgVariant: {
			control: 'select',
			options: ['background', 'secondary', 'card', 'default', 'muted', 'subtle'],
			description: 'Background variant',
			table: {
				type: { summary: "'background' | 'secondary' | 'card' | 'default' | 'muted' | 'subtle'" },
				defaultValue: { summary: 'background' },
				category: 'Appearance',
			},
		},
		align: {
			control: 'select',
			options: ['start', 'center'],
			description: 'Content alignment',
			table: {
				type: { summary: "'start' | 'center'" },
				defaultValue: { summary: 'center' },
				category: 'Layout',
			},
		},
		layout: {
			control: 'select',
			options: ['stack', 'split'],
			description: 'Header layout: stack or split (two-column on md+)',
			table: {
				type: { summary: "'stack' | 'split'" },
				defaultValue: { summary: 'stack' },
				category: 'Layout',
			},
		},
		paddingY: {
			control: 'select',
			options: ['lg', 'xl', '2xl'],
			description: 'Vertical padding size',
			table: {
				type: { summary: "'lg' | 'xl' | '2xl'" },
				defaultValue: { summary: '2xl' },
				category: 'Layout',
			},
		},
		maxWidth: {
			control: 'select',
			options: ['xl', '2xl', '3xl', '4xl', '5xl', '6xl', '7xl'],
			description: 'Maximum width of content',
			table: {
				type: { summary: "'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl'" },
				defaultValue: { summary: '4xl' },
				category: 'Layout',
			},
		},
		bleed: {
			control: 'boolean',
			description: 'Allow full-width background while constraining inner content',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Layout',
			},
		},
		divider: {
			control: 'boolean',
			description: 'Show a subtle separator under the header block',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof SectionContainer>

export const Default: Story = {
	args: {
		title: 'Section Title',
		description: 'This is a description that provides context about the section content.',
		children: (
			<div className="grid grid-cols-1 md:grid-cols-3 gap-6">
				<div className="p-6 bg-card rounded-lg border">
					<h3 className="font-semibold mb-2">Feature 1</h3>
					<p className="text-sm text-muted-foreground">Feature description</p>
				</div>
				<div className="p-6 bg-card rounded-lg border">
					<h3 className="font-semibold mb-2">Feature 2</h3>
					<p className="text-sm text-muted-foreground">Feature description</p>
				</div>
				<div className="p-6 bg-card rounded-lg border">
					<h3 className="font-semibold mb-2">Feature 3</h3>
					<p className="text-sm text-muted-foreground">Feature description</p>
				</div>
			</div>
		),
	},
}

export const WithBackground: Story = {
	args: {
		title: 'Section with Background',
		description: 'This section has a muted background to separate it from adjacent sections.',
		bgVariant: 'muted',
		children: (
			<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
				<div className="p-6 bg-card rounded-lg border">
					<h3 className="font-semibold mb-2">Card 1</h3>
					<p className="text-sm text-muted-foreground">Content with background</p>
				</div>
				<div className="p-6 bg-card rounded-lg border">
					<h3 className="font-semibold mb-2">Card 2</h3>
					<p className="text-sm text-muted-foreground">Content with background</p>
				</div>
			</div>
		),
	},
}

export const Narrow: Story = {
	args: {
		title: 'Narrow Section',
		description: 'This section uses a narrower max width for focused content.',
		maxWidth: '2xl',
		align: 'center',
		children: (
			<div className="prose prose-sm dark:prose-invert mx-auto">
				<p>
					This content is constrained to a narrower width, perfect for reading-focused sections like blog posts or
					documentation.
				</p>
			</div>
		),
	},
}

export const InOrganismContext: Story = {
	render: () => (
		<div className="w-full">
			<div className="mb-4 text-sm text-muted-foreground">Example: SectionContainer in HeroSection organism</div>
			<SectionContainer
				eyebrow="New Feature"
				title="Private LLM Hosting in the Netherlands"
				description="GDPR-compliant, self-hosted AI infrastructure for enterprises and developers."
				align="center"
				paddingY="2xl"
			>
				<div className="flex justify-center gap-4 mt-8">
					<button className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-colors">
						Get Started
					</button>
					<button className="px-6 py-3 bg-secondary text-secondary-foreground rounded-lg font-semibold hover:bg-secondary/80 transition-colors">
						Learn More
					</button>
				</div>
			</SectionContainer>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'SectionContainer as used in HeroSection. Provides centered alignment, eyebrow badge, and action buttons.',
			},
		},
	},
}
