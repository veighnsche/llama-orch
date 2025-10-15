import type { Meta, StoryObj } from '@storybook/react'
import { HeroSection } from './HeroSection'

const meta = {
	title: 'Organisms/HeroSection',
	component: HeroSection,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The HeroSection is the primary landing section of the rbee application. It features a compelling headline, value proposition, interactive terminal demo, and clear call-to-action buttons. The section uses a full-viewport height with animated elements and a honeycomb background pattern.

## Marketing Strategy

### Target Audience
Developers frustrated with cloud AI costs and vendor lock-in. Primary personas:
- Solo developers shipping SaaS with AI features
- Small teams burning $500+/mo on API fees
- DevOps engineers managing infrastructure
- Privacy-conscious developers requiring data residency

### Primary Message
**"AI Infrastructure. On Your Terms."** — Empowering developers to own their AI stack without cloud dependencies.

### Copy Analysis
- **Headline tone**: Direct, empowering, developer-focused
- **Emotional appeal**: Freedom from vendor lock-in, control over costs
- **Power words**: "Your hardware", "Your terms", "Zero API fees", "Drop-in"
- **Social proof**: GitHub stars, OpenAI compatibility, $0 cost

### Conversion Elements
- **Primary CTA**: "Get Started Free" — Action-oriented, removes friction (no payment)
- **Secondary CTA**: "View Docs" — Alternative path for technical validation
- **Trust signals**: 
  - "100% Open Source • GPL-3.0-or-later" badge
  - GitHub star link
  - "OpenAI-Compatible" API badge
  - "$0 • No Cloud Required" cost indicator
- **Interactive demo**: Terminal showing real GPU orchestration builds credibility

### Objection Handling
- **"Is it hard to set up?"** → Terminal shows simple commands
- **"Will it work with my tools?"** → "Drop-in OpenAI API" badge
- **"What's the catch?"** → "$0 • No Cloud Required"
- **"Is it production-ready?"** → GitHub stars + active community signal

### Variations to Test
- Alternative headline: "Run AI on Your Hardware. Keep 100% Control."
- Alternative CTA: "Install in 15 Minutes" (time-based urgency)
- Alternative positioning: Lead with cost savings vs. control/privacy

## Composition
This organism contains:
- **PulseBadge**: Animated badge showing open-source status
- **Headline**: Large, bold headline with primary color accent
- **Value Proposition**: Clear description of the product benefits
- **Feature Bullets**: Quick list of key benefits (checkmarks)
- **CTA Buttons**: Primary "Get Started" and secondary "View Docs" buttons
- **Trust Badges**: GitHub stars, API compatibility, cost indicators
- **TerminalWindow**: Interactive demo showing GPU orchestration
- **ProgressBar**: Visual GPU utilization indicators
- **FloatingKPICard**: Animated floating card with metrics
- **HoneycombPattern**: Background pattern for visual interest

## When to Use
- As the first section on the homepage
- To immediately communicate value proposition
- To provide interactive demo of the product
- To drive users to primary call-to-action

## Content Requirements
- **Headline**: Clear, benefit-focused headline (max 2 lines)
- **Subheadline**: Detailed value proposition (2-3 sentences)
- **Feature Bullets**: 3-5 key benefits
- **CTA Buttons**: Primary and secondary actions
- **Terminal Demo**: Realistic code example showing product in action
- **Trust Indicators**: Social proof elements (GitHub stars, etc.)

## Usage in Commercial Site

### Home Page (/)
\`\`\`tsx
<HeroSection />
\`\`\`

**Context**: First section, immediately visible on page load  
**Purpose**: Hook visitors within 3 seconds, communicate core value prop, drive to signup or docs  
**Metrics**: Primary conversion point for homepage traffic

## Variants
- **Default**: Full hero with all elements
- **Mobile**: Responsive layout with stacked content
- **Reduced Motion**: Respects prefers-reduced-motion setting

## Examples
\`\`\`tsx
import { HeroSection } from '@rbee/ui/organisms/HeroSection'

// Simple usage - no props needed
<HeroSection />
\`\`\`

## Related Components
- PulseBadge
- TerminalWindow
- ProgressBar
- FloatingKPICard
- HoneycombPattern

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **ARIA Labels**: Proper labels on all interactive elements
- **Semantic HTML**: Uses <section> with aria-labelledby
- **Reduced Motion**: Respects prefers-reduced-motion media query
- **Focus States**: Visible focus indicators on all interactive elements
- **Live Regions**: Terminal output uses aria-live for screen readers
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof HeroSection>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default hero section with all elements. Use the theme toggle in the toolbar to switch between light and dark modes. Use the viewport toolbar to test responsive behavior.',
			},
		},
	},
}

export const HomePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Exact implementation from \`/\` route.

**Marketing Notes:**
- Headline "AI Infrastructure. On Your Terms." establishes ownership/control theme
- Subheadline addresses specific pain: "avoid vendor lock-in"
- Three micro-proof bullets reduce friction: "Your GPUs", "Zero API fees", "Drop-in OpenAI API"
- Terminal demo shows real command (\`rbee-keeper infer\`) to build credibility
- GPU utilization bars demonstrate multi-GPU orchestration visually
- Cost counter "$0.00" reinforces zero-cost promise
- Trust badges provide social proof without overwhelming

**Conversion Strategy:**
- Primary CTA "Get Started Free" removes payment friction
- Secondary CTA "View Docs" serves technical validators
- No email capture at hero level (reduces friction)
- Open source badge builds trust with developer audience`,
			},
		},
	},
}

export const AlternativeHeadline: Story = {
	render: () => (
		<div>
			<HeroSection />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', textAlign: 'center' }}>
				<h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Alternative Headline Options</h3>
				<div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'left', lineHeight: '1.8' }}>
					<p><strong>Current:</strong> "AI Infrastructure. On Your Terms."</p>
					<p><strong>Alt 1:</strong> "Run AI on Your Hardware. Keep 100% Control." (More explicit about hardware ownership)</p>
					<p><strong>Alt 2:</strong> "Zero-Cost AI Infrastructure. No Vendor Lock-In." (Leads with cost savings)</p>
					<p><strong>Alt 3:</strong> "Private LLMs. Your Network. Your Rules." (Privacy-first angle)</p>
					<p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
						<strong>A/B Test Recommendation:</strong> Test Alt 2 for cost-conscious audience, Alt 3 for enterprise/privacy-focused.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Alternative headline variations for A/B testing. Each emphasizes different value propositions: control, cost, or privacy.',
			},
		},
	},
}

export const WithScrollIndicator: Story = {
	render: () => (
		<div>
			<HeroSection />
			<div style={{ padding: '4rem 2rem', textAlign: 'center', background: 'rgba(0,0,0,0.02)' }}>
				<h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '1rem' }}>Next Section</h2>
				<p>Scroll up to see the hero section with its full-viewport height and animations.</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Hero section with additional content below to demonstrate full-viewport height and scroll behavior.',
			},
		},
	},
}
