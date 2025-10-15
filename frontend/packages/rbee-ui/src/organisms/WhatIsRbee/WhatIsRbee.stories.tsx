import type { Meta, StoryObj } from '@storybook/react'
import { WhatIsRbee } from './WhatIsRbee'

const meta = {
	title: 'Organisms/WhatIsRbee',
	component: WhatIsRbee,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The WhatIsRbee section provides a comprehensive introduction to the rbee platform. It combines brand identity, value propositions, feature highlights, statistics, and a visual network diagram to communicate what rbee is and why it matters.

## Marketing Strategy

### Target Audience
New visitors who landed on the homepage and scrolled past the hero. They need:
- Clear explanation of what rbee actually is
- Proof that it's legitimate (stats, open source)
- Understanding of core benefits before diving into features

### Primary Message
**"Self-hosted LLM orchestration for the rest of us"** — Positioning rbee as accessible, not just for infrastructure experts.

### Copy Analysis
- **Headline tone**: Educational, approachable
- **Emotional appeal**: Independence from cloud providers, simplicity despite power
- **Power words**: "Open-source", "Self-hosted", "Independence", "Privacy", "Orchestration"
- **Social proof**: Stats showing adoption (GitHub stars, installations)

### Conversion Elements
- **Primary CTA**: "Get Started" — Continues conversion funnel from hero
- **Secondary CTA**: "See Architecture" — Serves technical validators
- **Value bullets**: Three key benefits (Independence, Privacy, GPU orchestration)
- **Stats grid**: Quantifies adoption and impact
- **Network diagram**: Visual proof of multi-GPU capability

### Objection Handling
- **"Is this just for experts?"** → "for the rest of us" positioning
- **"Is it proven?"** → Stats show 1,200+ stars, 500+ installations
- **"Does it actually work?"** → Network diagram shows real architecture
- **"Can I trust it?"** → "Open-source • Self-hosted" badge

### Variations to Test
- Alternative positioning: "Enterprise-grade orchestration. Homelab simplicity."
- Alternative stats: Highlight cost savings vs. adoption metrics
- Alternative CTA: "Install in 15 Minutes" vs. "Get Started"

## Composition

## Composition
This organism contains:
- **SectionContainer**: Wrapper with title and background variant
- **Badge**: "Open-source • Self-hosted" label
- **BrandMark + BrandWordmark**: Logo and wordmark with pronunciation tooltip
- **Value Bullets**: Three key benefits with IconBox components
- **StatsGrid**: Three stat cards showing key metrics
- **CTA Buttons**: Primary "Get Started" and secondary "See Architecture"
- **Network Diagram**: Visual showing distributed GPU orchestration
- **Technical Accent**: Architecture highlights

## When to Use
- On the homepage after the hero section
- As an introduction section on marketing pages
- To explain the platform to new visitors
- Before diving into detailed features

## Content Requirements
- **Headline**: Clear statement of what rbee is
- **Pronunciation**: Tooltip explaining "are-bee"
- **Description**: 2-3 sentence overview
- **Value Bullets**: 3 key benefits (Independence, Privacy, GPU orchestration)
- **Stats**: 3 compelling metrics
- **Visual**: Network diagram showing architecture
- **CTAs**: Primary and secondary actions

## Variants
- **Default**: Full section with all elements
- **Mobile**: Stacked layout with image below content

## Examples
\`\`\`tsx
import { WhatIsRbee } from '@rbee/ui/organisms/WhatIsRbee'

// Simple usage - no props needed
<WhatIsRbee />
\`\`\`

## Used In
- Home page (/)
- About page

## Related Components
- SectionContainer
- BrandMark
- BrandWordmark
- IconBox
- StatsGrid
- Badge

## Accessibility
- **Keyboard Navigation**: All buttons and links are keyboard accessible
- **Tooltips**: Pronunciation tooltip accessible via keyboard
- **Semantic HTML**: Proper heading hierarchy
- **Image Alt**: Detailed alt text for network diagram
- **Focus States**: Visible focus indicators on all interactive elements
- **Color Contrast**: Meets WCAG AA standards in both themes
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof WhatIsRbee>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default WhatIsRbee section with all elements. Use the theme toggle in the toolbar to switch between light and dark modes. Use the viewport toolbar to test responsive behavior.',
			},
		},
	},
}

export const HomePageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story: `**Home page context** — Appears immediately after HeroSection on \`/\` route.

**Marketing Notes:**
- **Brand introduction**: Logo + wordmark with pronunciation tooltip ("are-bee") removes confusion
- **Badge**: "Open-source • Self-hosted" establishes trust and positioning
- **Value bullets**: Three benefits address core concerns:
  1. Independence (no vendor lock-in)
  2. Privacy (data stays local)
  3. GPU orchestration (technical capability)
- **Stats grid**: Social proof through numbers (GitHub stars, installations, GPUs orchestrated, cost)
- **Network diagram**: Visual demonstration of distributed architecture

**Conversion Strategy:**
- Positioned after hero to educate visitors who didn't convert immediately
- Dual CTAs: "Get Started" (conversion) + "See Architecture" (technical validation)
- Stats build credibility without overwhelming
- Visual diagram makes complex concept tangible

**Tone**: Educational but accessible — "for the rest of us" positioning`,
			},
		},
	},
}

export const AlternativePositioning: Story = {
	render: () => (
		<div>
			<WhatIsRbee />
			<div style={{ padding: '2rem', background: 'rgba(0,0,0,0.02)', textAlign: 'center' }}>
				<h3 style={{ fontWeight: 'bold', marginBottom: '1rem' }}>Alternative Positioning Options</h3>
				<div style={{ maxWidth: '600px', margin: '0 auto', textAlign: 'left', lineHeight: '1.8' }}>
					<p><strong>Current:</strong> "Self-hosted LLM orchestration for the rest of us"</p>
					<p><strong>Alt 1:</strong> "Enterprise-grade orchestration. Homelab simplicity." (Contrast positioning)</p>
					<p><strong>Alt 2:</strong> "The open-source alternative to OpenAI and Anthropic" (Competitive positioning)</p>
					<p><strong>Alt 3:</strong> "Multi-GPU orchestration that just works" (Simplicity angle)</p>
					<p style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#666' }}>
						<strong>Recommendation:</strong> Test Alt 1 for enterprise audience, Alt 2 for cloud-frustrated developers.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Alternative positioning statements for A/B testing. Each targets different audience segments and pain points.',
			},
		},
	},
}
