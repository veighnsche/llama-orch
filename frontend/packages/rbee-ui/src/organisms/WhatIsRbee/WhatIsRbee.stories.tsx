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
