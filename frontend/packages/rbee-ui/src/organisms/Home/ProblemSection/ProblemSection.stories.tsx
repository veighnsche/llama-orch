import type { Meta, StoryObj } from '@storybook/react'
import { AlertTriangle, Cloud, DollarSign, Lock, Shield, TrendingDown } from 'lucide-react'
import { ProblemSection } from './ProblemSection'

const meta = {
	title: 'Organisms/Home/ProblemSection',
	component: ProblemSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The ProblemSection displays a grid of problem cards to highlight pain points and challenges. It uses visual hierarchy with icons, tone-based styling, optional loss tags, and a CTA banner to drive action.

## Marketing Strategy

### Target Audience
Visitors who landed on the homepage and need to understand the problem before evaluating the solution. They need:
- Clear articulation of pain points
- Validation that their problems are understood
- Urgency to act (quantified losses)
- Emotional connection to frustrations

### Primary Message
Implicit: **"We understand your pain"** — Empathy-driven problem framing.

### Copy Analysis
- **Headline tone**: Problem-focused, empathetic
- **Emotional appeal**: Frustration with status quo, urgency to change
- **Power words**: "Unpredictable", "lock-in", "risk", "concerns", "Loss"
- **Social proof**: Loss tags quantify impact ("Loss €50/mo")

### Conversion Elements
- **Three problem cards**: Cost, vendor lock-in, privacy
- **Icons**: Visual differentiation (DollarSign, Lock, Shield)
- **Tone-based styling**: Destructive (red) for critical, primary (blue) for important
- **Loss tags**: Quantify monthly cost impact
- **Optional CTA banner**: Drive to solution after problem framing

### Problem Framing
1. **Unpredictable API costs**: Usage-based pricing scales unpredictably
2. **Vendor lock-in risk**: Proprietary APIs create dependencies
3. **Privacy & compliance concerns**: Data leaves your network

### Objection Handling
- **"Is this really a problem?"** → Loss tags quantify impact
- **"Can't I just use free tier?"** → Unpredictable costs, rate limits
- **"Is vendor lock-in that bad?"** → Risk framing, dependency concerns
- **"Do I need to worry about privacy?"** → Compliance concerns, audit requirements

### Variations to Test
- Alternative problem order: Lead with cost vs. privacy vs. lock-in
- Alternative tone: More aggressive (fear) vs. neutral (facts)
- Alternative loss tags: Monthly vs. annual vs. percentage

## Composition

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
import { ProblemSection } from '@rbee/ui/organisms'
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
					'Default problem section with three problems and CTA banner. Use the theme toggle in the toolbar to switch between light and dark modes. Use the viewport toolbar to test responsive behavior.',
			},
		},
	},
}

export const HomePageContext: Story = {
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- **Three problems**: Unpredictable costs, Vendor lock-in, Privacy concerns
- **Visual hierarchy**: Icons + tone-based styling
- **Optional CTA**: Drives to solution after problem framing

**Problem 1: Unpredictable API costs**
- **Icon**: DollarSign (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "Usage-based pricing scales unpredictably. What starts at $20/month becomes $2000/month."
- **Tag**: "Loss €50/mo" (quantified impact)
- **Target**: Cost-conscious developers, startups with budget pressure

**Problem 2: Vendor lock-in risk**
- **Icon**: Lock (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "Your entire codebase depends on proprietary APIs. Switching providers means rewriting everything."
- **Tag**: "High risk" (qualitative impact)
- **Target**: Developers concerned about dependencies, long-term maintainability

**Problem 3: Privacy & compliance concerns**
- **Icon**: Shield (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "Your code and prompts are sent to external servers. Compliance becomes a nightmare."
- **No tag**: (impact is qualitative)
- **Target**: Enterprise, regulated industries, privacy-conscious developers

**Conversion Strategy:**
- Three problems cover main pain points (cost, lock-in, privacy)
- Tone-based styling creates urgency (red = critical)
- Loss tags quantify impact (not abstract)
- Optional CTA drives to solution section

**Tone**: Empathetic, urgent, problem-focused`,
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
				icon: <Lock className="h-6 w-6" />,
				tone: 'destructive' as const,
				tag: 'High risk',
			},
			{
				title: 'Unpredictable costs',
				body: 'Usage-based pricing scales exponentially. What starts at $20/month becomes $2000/month.',
				icon: <TrendingDown className="h-6 w-6" />,
				tone: 'primary' as const,
				tag: 'Loss €180/mo',
			},
			{
				title: 'Data privacy concerns',
				body: 'Your code and prompts are sent to external servers. Compliance becomes a nightmare.',
				icon: <Shield className="h-6 w-6" />,
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

export const ToneVariations: Story = {
	args: {
		title: 'Problem tone variations',
		subtitle: 'Different visual treatments for different problem types',
		items: [
			{
				title: 'Destructive tone',
				body: 'Used for critical problems and risks. Red/destructive color scheme.',
				icon: <AlertTriangle className="h-6 w-6" />,
				tone: 'destructive' as const,
				tag: 'Critical',
			},
			{
				title: 'Primary tone',
				body: 'Used for important problems related to cost or efficiency. Primary color scheme.',
				icon: <DollarSign className="h-6 w-6" />,
				tone: 'primary' as const,
				tag: 'Important',
			},
			{
				title: 'Muted tone',
				body: 'Used for less urgent problems or informational items. Neutral color scheme.',
				icon: <Cloud className="h-6 w-6" />,
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
