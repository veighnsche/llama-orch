import type { Meta, StoryObj } from '@storybook/react'
import { TestimonialsRail } from './TestimonialsRail'

const meta = {
	title: 'Organisms/TestimonialsRail',
	component: TestimonialsRail,
	parameters: {
		layout: 'fullwidth',
		docs: {
			description: {
				component: `
## Overview
The TestimonialsRail component displays customer testimonials in a grid or carousel layout, with optional stats section. It filters testimonials by sector and supports different layout modes.

## Composition
This organism is composed of:
- **TestimonialCard**: Individual testimonial cards with quote, author, role, and avatar (molecule)
- **StatsGrid**: Optional statistics display showing key metrics (molecule)
- **Layout Container**: Grid or carousel wrapper with responsive behavior

## When to Use
- On the Developers page to show developer testimonials
- On the Providers page to show provider testimonials
- On the Enterprise page to show enterprise customer testimonials
- In the home page social proof section
- Anywhere you need to display customer quotes and social proof

## Variants
- **Grid Layout**: Standard grid display (default on desktop)
- **Carousel Layout**: Horizontal scrolling on mobile, grid on desktop
- **With Stats**: Includes statistics below testimonials
- **Without Stats**: Testimonials only
- **Sector Filtered**: Show testimonials from specific sectors (developers, providers, enterprise)

## Used In Commercial Site
- **Developers Page**: Developer testimonials with stats, sector filter: 'developers'
- **Providers Page**: Provider testimonials, sector filter: 'providers'
- **Enterprise Page**: Enterprise testimonials, sector filter: 'enterprise'
- **Home Page**: Mixed testimonials from all sectors
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		sectorFilter: {
			control: 'select',
			options: ['developers', 'providers', 'enterprise', undefined],
			description: 'Filter testimonials by sector. Can be a single sector or array of sectors.',
			table: {
				type: { summary: "Sector | Sector[] | undefined" },
				defaultValue: { summary: 'undefined' },
				category: 'Content',
			},
		},
		limit: {
			control: 'number',
			description: 'Maximum number of testimonials to display',
			table: {
				type: { summary: 'number' },
				defaultValue: { summary: 'undefined' },
				category: 'Content',
			},
		},
		layout: {
			control: 'select',
			options: ['grid', 'carousel'],
			description: 'Layout mode: grid (standard) or carousel (horizontal scroll on mobile)',
			table: {
				type: { summary: "'grid' | 'carousel'" },
				defaultValue: { summary: 'grid' },
				category: 'Appearance',
			},
		},
		showStats: {
			control: 'boolean',
			description: 'Whether to show statistics section below testimonials',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Content',
			},
		},
		className: {
			control: 'text',
			description: 'Additional CSS classes',
			table: {
				type: { summary: 'string' },
				category: 'Appearance',
			},
		},
		headingId: {
			control: 'text',
			description: 'ID for the heading element (for accessibility)',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'testimonials-h2' },
				category: 'Accessibility',
			},
		},
	},
} satisfies Meta<typeof TestimonialsRail>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		layout: 'grid',
		showStats: false,
		limit: 3,
	},
}

export const DevelopersPageDefault: Story = {
	args: {
		sectorFilter: 'developers',
		layout: 'grid',
		showStats: true,
		limit: 3,
	},
	parameters: {
		docs: {
			description: {
				story: 'TestimonialsRail as used on the Developers page. Shows developer testimonials with stats, filtered by sector.',
			},
		},
	},
}

export const ProvidersPageDefault: Story = {
	args: {
		sectorFilter: 'providers',
		layout: 'grid',
		showStats: true,
		limit: 3,
	},
	parameters: {
		docs: {
			description: {
				story: 'TestimonialsRail as used on the Providers page. Shows provider testimonials with stats about earnings and uptime.',
			},
		},
	},
}

export const EnterprisePageDefault: Story = {
	args: {
		sectorFilter: 'enterprise',
		layout: 'grid',
		showStats: false,
		limit: 3,
	},
	parameters: {
		docs: {
			description: {
				story: 'TestimonialsRail as used on the Enterprise page. Shows enterprise customer testimonials, no stats.',
			},
		},
	},
}

export const WithoutStats: Story = {
	args: {
		layout: 'grid',
		showStats: false,
		limit: 3,
	},
	parameters: {
		docs: {
			description: {
				story: 'Testimonials only, without the stats section. Useful when stats are shown elsewhere or not needed.',
			},
		},
	},
}

export const StatsOnly: Story = {
	args: {
		layout: 'grid',
		showStats: true,
		limit: 0,
	},
	parameters: {
		docs: {
			description: {
				story: 'Stats rail only, without testimonials. Note: This is a contrived example - in practice you would use StatsGrid directly.',
			},
		},
	},
}

export const CarouselLayout: Story = {
	args: {
		layout: 'carousel',
		showStats: false,
		limit: 6,
	},
	parameters: {
		docs: {
			description: {
				story: 'Carousel layout with horizontal scrolling on mobile. On desktop, displays as a grid. Try resizing your browser to see the responsive behavior.',
			},
		},
	},
}

export const AllSectors: Story = {
	args: {
		sectorFilter: undefined,
		layout: 'grid',
		showStats: true,
		limit: 6,
	},
	parameters: {
		docs: {
			description: {
				story: 'All testimonials from all sectors (developers, providers, enterprise). Useful for home page or general social proof.',
			},
		},
	},
}
