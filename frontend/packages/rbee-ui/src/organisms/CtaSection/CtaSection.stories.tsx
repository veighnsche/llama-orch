import type { Meta, StoryObj } from '@storybook/react'
import { ArrowRight, Download, Github } from 'lucide-react'
import { CTASection } from './CtaSection'

const meta = {
	title: 'Organisms/CTASection',
	component: CTASection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The CTASection is a flexible call-to-action component designed to drive user engagement at key points in the user journey. It features a headline, optional subtitle, primary and secondary action buttons, and optional trust messaging.

## Composition
This organism contains:
- **Eyebrow**: Optional small badge/label above title
- **Title**: Large, bold headline
- **Subtitle**: Optional supporting description
- **Primary Action**: Main CTA button with optional icons
- **Secondary Action**: Optional secondary button
- **Trust Note**: Optional small text for reassurance
- **Background Emphasis**: Optional gradient background effect

## When to Use
- At the end of feature sections to drive conversion
- On landing pages to encourage signup/download
- In the middle of long-form content to re-engage users
- After testimonials or social proof sections

## Content Requirements
- **Title**: Clear, action-oriented headline (max 2 lines)
- **Subtitle**: Optional context or benefit statement
- **Primary CTA**: Clear action verb (e.g., "Get Started", "Download Now")
- **Secondary CTA**: Optional alternative action
- **Trust Note**: Optional reassurance (e.g., "No credit card required")

## Variants
- **Default**: Centered layout without background emphasis
- **Left-Aligned**: Left-aligned text and buttons
- **With Gradient**: Subtle radial gradient background
- **Single Button**: Primary action only
- **Two Buttons**: Primary and secondary actions

## Examples
\`\`\`tsx
import { CTASection } from '@rbee/ui/organisms/CtaSection'
import { ArrowRight } from 'lucide-react'

// Basic CTA
<CTASection
  title="Ready to get started?"
  subtitle="Join thousands of developers running AI on their own hardware"
  primary={{
    label: "Get Started Free",
    href: "/signup",
    iconRight: ArrowRight
  }}
  secondary={{
    label: "View Documentation",
    href: "/docs"
  }}
  note="No credit card required"
/>

// With gradient emphasis
<CTASection
  eyebrow="Limited Time Offer"
  title="Start your free trial today"
  primary={{
    label: "Start Free Trial",
    href: "/trial"
  }}
  emphasis="gradient"
/>
\`\`\`

## Used In
- Home page (multiple sections)
- Feature pages
- Pricing page
- Documentation pages

## Related Components
- Button
- Link (Next.js)

## Accessibility
- **Keyboard Navigation**: All buttons are keyboard accessible
- **Focus States**: Visible focus indicators on all interactive elements
- **Semantic HTML**: Uses <section> element
- **Motion**: Respects prefers-reduced-motion for animations
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		title: {
			control: 'text',
			description: 'Main headline text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		subtitle: {
			control: 'text',
			description: 'Optional supporting text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		eyebrow: {
			control: 'text',
			description: 'Optional small label above title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		note: {
			control: 'text',
			description: 'Optional trust message below buttons',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		align: {
			control: 'select',
			options: ['center', 'left'],
			description: 'Text and button alignment',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'center' },
				category: 'Appearance',
			},
		},
		emphasis: {
			control: 'select',
			options: ['none', 'gradient'],
			description: 'Background emphasis style',
			table: {
				type: { summary: 'string' },
				defaultValue: { summary: 'none' },
				category: 'Appearance',
			},
		},
	},
} satisfies Meta<typeof CTASection>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	args: {
		title: 'Ready to run AI on your own terms?',
		subtitle: 'Join thousands of developers who have taken control of their AI infrastructure',
		primary: {
			label: 'Get Started Free',
			href: '/signup',
			iconRight: ArrowRight,
		},
		secondary: {
			label: 'View Documentation',
			href: '/docs',
		},
		note: 'No credit card required • Cancel anytime',
	},
}

export const SingleButton: Story = {
	args: {
		title: 'Download rbee today',
		subtitle: 'Start running LLMs on your hardware in minutes',
		primary: {
			label: 'Download Now',
			href: '/download',
			iconLeft: Download,
		},
		note: 'Free and open source • GPL-3.0 license',
	},
	parameters: {
		docs: {
			description: {
				story: 'CTA with only a primary button and no secondary action.',
			},
		},
	},
}

export const WithGradient: Story = {
	args: {
		eyebrow: 'Limited Time Offer',
		title: 'Start your 30-day free trial',
		subtitle: 'Experience the full power of rbee with no limitations',
		primary: {
			label: 'Start Free Trial',
			href: '/trial',
			iconRight: ArrowRight,
		},
		secondary: {
			label: 'See Pricing',
			href: '/pricing',
		},
		emphasis: 'gradient',
		note: 'No credit card required',
	},
	parameters: {
		docs: {
			description: {
				story: 'CTA with gradient background emphasis for extra visual impact.',
			},
		},
	},
}

export const LeftAligned: Story = {
	args: {
		title: 'Join the rbee community',
		subtitle: 'Connect with developers, share knowledge, and contribute to the project',
		primary: {
			label: 'Join Discord',
			href: 'https://discord.gg/rbee',
		},
		secondary: {
			label: 'View on GitHub',
			href: 'https://github.com/veighnsche/llama-orch',
			iconLeft: Github,
		},
		align: 'left',
	},
	parameters: {
		docs: {
			description: {
				story: 'Left-aligned variant, useful for asymmetric layouts or when paired with images.',
			},
		},
	},
}

export const MinimalWithEyebrow: Story = {
	args: {
		eyebrow: 'Open Source',
		title: 'Contribute to rbee',
		primary: {
			label: 'View Repository',
			href: 'https://github.com/veighnsche/llama-orch',
			iconRight: ArrowRight,
		},
	},
	parameters: {
		docs: {
			description: {
				story: 'Minimal CTA with eyebrow badge and single action.',
			},
		},
	},
}

export const MobileView: Story = {
	args: {
		title: 'Get started with rbee',
		subtitle: 'Run LLMs on your own hardware',
		primary: {
			label: 'Download Free',
			href: '/download',
		},
		secondary: {
			label: 'Learn More',
			href: '/docs',
		},
	},
	parameters: {
		viewport: {
			defaultViewport: 'mobile1',
		},
		docs: {
			description: {
				story: 'Mobile view with stacked buttons.',
			},
		},
	},
}

export const AllVariants: Story = {
	render: () => (
		<div>
			<CTASection
				title="Centered with gradient"
				subtitle="Default centered alignment with gradient background"
				primary={{ label: 'Primary Action', href: '#' }}
				secondary={{ label: 'Secondary', href: '#' }}
				emphasis="gradient"
			/>
			<CTASection
				title="Left-aligned without gradient"
				subtitle="Left alignment for asymmetric layouts"
				primary={{ label: 'Primary Action', href: '#' }}
				secondary={{ label: 'Secondary', href: '#' }}
				align="left"
			/>
			<CTASection
				eyebrow="With Eyebrow"
				title="Single button with badge"
				primary={{ label: 'Single Action', href: '#', iconRight: ArrowRight }}
				note="Trust message below button"
			/>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'All CTA variants displayed together for comparison.',
			},
		},
	},
}
