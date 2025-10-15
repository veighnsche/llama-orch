import type { Meta, StoryObj } from '@storybook/react'
import { PricingSection } from './PricingSection'

const meta = {
	title: 'Organisms/PricingSection',
	component: PricingSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The PricingSection displays pricing tiers for the rbee platform with a monthly/yearly toggle, feature lists, and call-to-action buttons. It includes trust badges, an optional editorial image, and footer reassurance text.

## Composition
This organism contains:
- **SectionContainer**: Wrapper with title and description
- **Kicker Badges**: Feature highlights (Open source, OpenAI-compatible, etc.)
- **Billing Toggle**: Monthly/yearly switcher with savings badge
- **PricingTier Cards**: Three tiers (Home/Lab, Team, Enterprise)
- **Editorial Image**: Optional visual showing product progression
- **Footer Reassurance**: Trust messaging and pricing notes

## When to Use
- On the pricing page as the main content
- On the homepage to show pricing overview
- In marketing materials to communicate value

## Content Requirements
- **Section Title**: Clear headline about pricing
- **Description**: Brief explanation of pricing philosophy
- **Pricing Tiers**: 3 tiers with features and CTAs
- **Trust Badges**: Key features (open source, no feature gates)
- **Footer Text**: Reassurance and legal disclaimers

## Variants
- **Home**: Condensed version for homepage
- **Pricing**: Full version for dedicated pricing page
- **With/Without Image**: Toggle editorial image
- **With/Without Footer**: Toggle reassurance text
- **With/Without Kicker**: Toggle feature badges

## Examples
\`\`\`tsx
import { PricingSection } from '@rbee/ui/organisms/PricingSection'

// Homepage variant
<PricingSection variant="home" />

// Pricing page variant
<PricingSection 
  variant="pricing"
  title="Simple, honest pricing"
  description="Every plan includes the full orchestrator"
/>

// Minimal variant
<PricingSection
  showKicker={false}
  showEditorialImage={false}
  showFooter={false}
/>
\`\`\`

## Used In
- Home page (/)
- Pricing page (/pricing)
- Developers page (/developers)

## Related Components
- SectionContainer
- PricingTier
- Button

## Accessibility
- **Keyboard Navigation**: All buttons and toggle are keyboard accessible
- **ARIA**: Billing toggle uses aria-pressed
- **Focus States**: Visible focus indicators on all interactive elements
- **Semantic HTML**: Proper heading hierarchy
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		variant: {
			control: 'select',
			options: ['home', 'pricing'],
			description: 'Variant for different page contexts',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'home' },
				category: 'Appearance',
			},
		},
		title: {
			control: 'text',
			description: 'Override section title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		description: {
			control: 'text',
			description: 'Override section description',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		showKicker: {
			control: 'boolean',
			description: 'Show feature badges above title',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Appearance',
			},
		},
		showEditorialImage: {
			control: 'boolean',
			description: 'Show editorial image below pricing cards',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Appearance',
			},
		},
		showFooter: {
			control: 'boolean',
			description: 'Show footer reassurance text',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Appearance',
			},
		},
	},
} satisfies Meta<typeof PricingSection>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		variant: 'home',
	},
	parameters: {
		docs: {
			description: {
				story:
					'Default pricing section for homepage. Use the theme toggle in the toolbar to switch between light and dark modes.',
			},
		},
	},
}

export const PricingPage: Story = {
	args: {
		variant: 'pricing',
	},
	parameters: {
		docs: {
			description: {
				story: 'Full pricing section variant for dedicated pricing page with different copy.',
			},
		},
	},
}

export const MinimalVariant: Story = {
	args: {
		variant: 'home',
		showKicker: false,
		showEditorialImage: false,
		showFooter: false,
	},
	parameters: {
		docs: {
			description: {
				story: 'Minimal variant with only pricing cards, no badges, image, or footer.',
			},
		},
	},
}

export const WithoutImage: Story = {
	args: {
		variant: 'pricing',
		showEditorialImage: false,
	},
	parameters: {
		docs: {
			description: {
				story: 'Pricing section without the editorial image, useful for tighter layouts.',
			},
		},
	},
}

export const CustomContent: Story = {
	args: {
		variant: 'home',
		title: 'Choose your plan',
		description: 'Start free, upgrade when you need more power and collaboration features',
	},
	parameters: {
		docs: {
			description: {
				story: 'Pricing section with custom title and description.',
			},
		},
	},
}

export const InteractiveBillingToggle: Story = {
	render: () => (
		<div>
			<PricingSection variant="pricing" />
			<div style={{ padding: '2rem', textAlign: 'center', background: 'rgba(0,0,0,0.02)' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Try the Billing Toggle</h2>
				<ol
					style={{
						listStyle: 'decimal',
						paddingLeft: '2rem',
						textAlign: 'left',
						maxWidth: '600px',
						margin: '0 auto',
						lineHeight: '1.8',
					}}
				>
					<li>Click the "Monthly" or "Yearly" button above</li>
					<li>Watch the Team tier price update</li>
					<li>Notice the "Save 2 months" badge on yearly</li>
					<li>Prices for Home/Lab (free) and Enterprise (custom) don't change</li>
				</ol>
				<p style={{ marginTop: '1rem', color: '#666' }}>The toggle uses aria-pressed for accessibility.</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Interactive demo of the monthly/yearly billing toggle.',
			},
		},
	},
}

export const PricingFeatures: Story = {
	render: () => (
		<div>
			<PricingSection variant="pricing" />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem', textAlign: 'center' }}>
					Pricing Highlights
				</h2>
				<div
					style={{
						maxWidth: '800px',
						margin: '0 auto',
						display: 'grid',
						gap: '1rem',
						gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
					}}
				>
					<div>
						<h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Home/Lab (Free)</h3>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Unlimited GPUs on your hardware. No feature gates. Perfect for individuals and homelabs.
						</p>
					</div>
					<div>
						<h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Team (€99/mo)</h3>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Web UI, shared workspaces, priority support. Most popular for growing teams.
						</p>
					</div>
					<div>
						<h3 style={{ fontWeight: 'bold', marginBottom: '0.5rem' }}>Enterprise (Custom)</h3>
						<p style={{ fontSize: '0.875rem', color: '#666' }}>
							Dedicated instances, custom SLAs, white-label options. For organizations.
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Overview of the three pricing tiers and their key differentiators.',
			},
		},
	},
}
