import type { Meta, StoryObj } from '@storybook/react'
import { AlertTriangle, Cloud, DollarSign, Lock, Shield, TrendingDown } from 'lucide-react'
import { ProblemSection } from './ProblemSection'

const meta = {
	title: 'Organisms/ProblemSection',
	component: ProblemSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProblemSection displays a grid of problem cards to highlight pain points and challenges. It uses visual hierarchy with icons, tone-based styling, optional loss tags, and a CTA banner to drive action.

## Composition
This organism contains:
- **Header**: Optional kicker, title, and subtitle
- **Problem Cards**: Grid of 3 cards with:
  - Icon (component or ReactNode)
  - Title and body text
  - Optional tag (e.g., "Loss €50/mo")
  - Tone-based styling (primary, destructive, muted)
- **CTA Banner**: Optional call-to-action with primary and secondary buttons
- **Staggered Animations**: Cards animate in with delays

## When to Use
- To highlight problems that the product solves
- Before solution or features sections
- On landing pages to create urgency
- To establish pain points before presenting value

## Content Requirements
- **Title**: Clear problem statement
- **Subtitle**: Context or amplification
- **Problem Items**: 3 cards with titles, descriptions, icons
- **Tags**: Optional loss indicators or metrics
- **CTA**: Clear next action

## Variants
- **Default**: Three problems with destructive/primary tones
- **Custom Problems**: Override with custom items
- **With CTA**: Include call-to-action banner
- **Without CTA**: Problems only

## Examples
\`\`\`tsx
import { ProblemSection } from '@rbee/ui/organisms/ProblemSection'
import { AlertTriangle, DollarSign, Lock } from 'lucide-react'

// Default usage
<ProblemSection />

// Custom problems
<ProblemSection
  title="Why developers struggle with AI"
  subtitle="Common challenges when building with AI"
  items={[
    {
      title: 'Vendor lock-in',
      body: 'Your code depends on proprietary APIs',
      icon: Lock,
      tone: 'destructive',
      tag: 'High risk'
    },
    {
      title: 'Rising costs',
      body: 'API fees multiply with team size',
      icon: DollarSign,
      tone: 'primary',
      tag: 'Loss €200/mo'
    },
    {
      title: 'Model changes',
      body: 'Updates break your workflows',
      icon: AlertTriangle,
      tone: 'destructive'
    }
  ]}
  ctaPrimary={{ label: 'Start Free', href: '/signup' }}
  ctaSecondary={{ label: 'Learn More', href: '/docs' }}
  ctaCopy="Take control of your AI infrastructure"
/>
\`\`\`

## Used In
- Home page (/)
- Landing pages
- Marketing pages

## Related Components
- Button
- IconBox

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **Focus States**: Visible focus indicators
- **Semantic HTML**: Proper heading hierarchy
- **Motion**: Respects prefers-reduced-motion
- **Color Contrast**: Meets WCAG AA standards in both themes
- **ARIA**: Icons marked as aria-hidden
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		title: {
			control: 'text',
			description: 'Section title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		subtitle: {
			control: 'text',
			description: 'Section subtitle',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		kicker: {
			control: 'text',
			description: 'Small text above title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
	},
} satisfies Meta<typeof ProblemSection>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default problem section with three problems and CTA banner. Use the theme toggle in the toolbar to switch between light and dark modes.',
			},
		},
	},
}

export const WithoutCTA: Story = {
	args: {
		ctaPrimary: undefined,
		ctaSecondary: undefined,
		ctaCopy: undefined,
	},
	parameters: {
		docs: {
			description: {
				story: 'Problem section without CTA banner, showing only the problem cards.',
			},
		},
	},
}

export const CustomProblems: Story = {
	args: {
		kicker: 'Common Challenges',
		title: 'Why cloud AI is risky',
		subtitle: "Relying on external providers creates dependencies you can't control",
		items: [
			{
				title: 'Vendor lock-in',
				body: 'Your entire codebase depends on proprietary APIs. Switching providers means rewriting everything.',
				icon: Lock,
				tone: 'destructive' as const,
				tag: 'High risk',
			},
			{
				title: 'Unpredictable costs',
				body: 'Usage-based pricing scales exponentially. What starts at $20/month becomes $2000/month.',
				icon: TrendingDown,
				tone: 'primary' as const,
				tag: 'Loss €180/mo',
			},
			{
				title: 'Data privacy concerns',
				body: 'Your code and prompts are sent to external servers. Compliance becomes a nightmare.',
				icon: Shield,
				tone: 'destructive' as const,
			},
		],
		ctaPrimary: { label: 'Host Your Own', href: '/signup' },
		ctaSecondary: { label: 'Compare Options', href: '/pricing' },
		ctaCopy: 'Run AI on your own infrastructure and eliminate these risks',
	},
	parameters: {
		docs: {
			description: {
				story: 'Custom problem items with different icons, tones, and tags.',
			},
		},
	},
}

export const MobileView: Story = {
	parameters: {
		viewport: {
			defaultViewport: 'mobile1',
		},
		docs: {
			description: {
				story: 'Mobile view with stacked single-column layout.',
			},
		},
	},
}

export const TabletView: Story = {
	parameters: {
		viewport: {
			defaultViewport: 'tablet',
		},
		docs: {
			description: {
				story: 'Tablet view showing responsive grid behavior.',
			},
		},
	},
}

export const ToneVariations: Story = {
	args: {
		title: 'Problem tone variations',
		subtitle: 'Different visual treatments for different problem types',
		items: [
			{
				title: 'Destructive tone',
				body: 'Used for critical problems and risks. Red/destructive color scheme.',
				icon: AlertTriangle,
				tone: 'destructive' as const,
				tag: 'Critical',
			},
			{
				title: 'Primary tone',
				body: 'Used for important problems related to cost or efficiency. Primary color scheme.',
				icon: DollarSign,
				tone: 'primary' as const,
				tag: 'Important',
			},
			{
				title: 'Muted tone',
				body: 'Used for less urgent problems or informational items. Neutral color scheme.',
				icon: Cloud,
				tone: 'muted' as const,
			},
		],
	},
	parameters: {
		docs: {
			description: {
				story: 'Demonstrates the three tone options: destructive, primary, and muted.',
			},
		},
	},
}
