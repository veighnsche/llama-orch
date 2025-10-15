import type { Meta, StoryObj } from '@storybook/react'
import { ArrowRight, Download, Github } from 'lucide-react'
import { CTASection } from './CtaSection'

const meta = {
	title: 'Organisms/Home/CTASection',
	component: CTASection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The CTASection is a flexible call-to-action component designed to drive user engagement at key points in the user journey. It features a headline, optional subtitle, primary and secondary action buttons, and optional trust messaging.

## Marketing Strategy

### Target Audience
Visitors at decision points throughout the site. They need:
- Clear next action (no ambiguity)
- Removal of friction ("No credit card required")
- Urgency or motivation to act now
- Alternative path if not ready for primary action

### Primary Message
Context-dependent, but generally: **"Take action now"** — Direct, action-oriented.

### Copy Analysis
- **Headline tone**: Action-oriented, urgent
- **Emotional appeal**: FOMO (don't miss out), empowerment (take control)
- **Power words**: "Stop", "Start", "Free", "Today", "Join"
- **Social proof**: "Join 500+ developers" (implicit validation)

### Conversion Elements
- **Headline**: Action-oriented statement
- **Subtitle**: Supporting context or benefit
- **Primary CTA**: Main action (Get Started, Download, etc.)
- **Secondary CTA**: Alternative path (View Docs, See Pricing, etc.)
- **Trust note**: Friction removal ("No credit card required", "Cancel anytime")
- **Optional eyebrow**: Urgency indicator ("Limited Time Offer")
- **Optional gradient**: Visual emphasis

### CTA Strategy by Context
- **Home page**: "Get Started Free" + "View Documentation"
- **Developers page**: "Install in 15 Minutes" + "See Examples"
- **Pricing page**: "Start Free Trial" + "Contact Sales"
- **Features page**: "Try It Now" + "Read Architecture"

### Objection Handling
- **"What's the commitment?"** → "No credit card required"
- **"Can I cancel?"** → "Cancel anytime"
- **"Is it really free?"** → "100% open source"
- **"What if I need help?"** → Secondary CTA to docs/support

### Variations to Test
- Alternative headlines: Action-oriented vs. benefit-oriented
- Alternative primary CTA: "Get Started" vs. "Install Now" vs. "Try Free"
- Alternative trust notes: Emphasize free vs. no commitment vs. open source

## Composition

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

export const HomePageContext: Story = {
	args: {
		title: 'Stop depending on AI providers. Start building today.',
		subtitle: "Join 500+ developers who've taken control of their AI infrastructure.",
		primary: {
			label: 'Get started free',
			href: '/getting-started',
			iconRight: ArrowRight,
		},
		secondary: {
			label: 'View documentation',
			href: '/docs',
			variant: 'outline' as const,
		},
		note: '100% open source. No credit card required. Install in 15 minutes.',
		emphasis: 'gradient' as const,
	},
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Headline**: "Stop depending on AI providers. Start building today." — Action-oriented, contrasting verbs
- **Subtitle**: "Join 500+ developers who've taken control..." — Social proof
- **Primary CTA**: "Get started free" — Action + friction removal
- **Secondary CTA**: "View documentation" — Alternative path for validators
- **Trust note**: Three friction removers:
  1. "100% open source" (transparency)
  2. "No credit card required" (no commitment)
  3. "Install in 15 minutes" (time-specific)
- **Emphasis**: Gradient background for visual impact

**Conversion Strategy:**
- Positioned at end of homepage (final conversion point)
- Headline uses "Stop/Start" contrast (action-oriented)
- Social proof ("500+ developers") builds confidence
- Primary CTA removes friction ("free", "no credit card")
- Secondary CTA serves technical validators
- Trust note stacks three friction removers
- Gradient emphasis draws attention

**Tone**: Urgent, action-oriented, empowering`,
			},
		},
	},
}

export const DevelopersPageContext: Story = {
	args: {
		title: 'Start building with rbee',
		subtitle: 'Install in 15 minutes. Use with your existing tools.',
		primary: {
			label: 'Get Started',
			href: '/getting-started',
			iconRight: ArrowRight,
		},
		secondary: {
			label: 'View Examples',
			href: '/docs/examples',
		},
		note: 'OpenAI-compatible API. Works with Zed, Cursor, Continue.',
	},
	parameters: {
		docs: {
			description: {
				story: `**Developers page context** — Variant used on \`/developers\` page.

**Differences from home page:**
- Simpler headline ("Start building with rbee")
- Time-specific subtitle ("Install in 15 minutes")
- Tool compatibility in trust note
- No gradient emphasis (cleaner)
- Secondary CTA to examples (more technical)

**Use case**: Developers page where visitors are already technical and need practical next steps.`,
			},
		},
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

export const AllVariants: Story = {
	args: {
		title: 'All Variants',
		primary: { label: 'Primary', href: '#' },
	},
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
