import type { Meta, StoryObj } from '@storybook/react'
import { UseCasesHero } from './UseCasesHero'

const meta = {
	title: 'Organisms/UseCases/UseCasesHero',
	component: UseCasesHero,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The UseCasesHero is the hero section for the Use Cases page, featuring a headline focused on independence and control, audience chips for quick navigation (Developers, Enterprise, Homelab), and a visual showcasing a homelab setup. It emphasizes the "your hardware, your rules" value proposition.

## Use Case Storytelling

### Narrative Structure
1. **Persona:** "Those Who Value Independence" - broad appeal across personas
2. **Scenario:** Need AI infrastructure without compromising power or flexibility
3. **Solution:** rbee adapts to your needs (solo developers to enterprises)
4. **Outcome:** Own your AI infrastructure with full control

### Emotional Journey
- **Before:** Vendor lock-in, lack of control, compromised flexibility
- **After:** Independence, ownership, full control over AI infrastructure

### Target Personas (Quick Navigation)
- **Developers:** Solo developers and small teams
- **Enterprise:** Organizations with compliance requirements
- **Homelab:** Enthusiasts with existing hardware

## Composition
This organism contains:
- **Badge**: "OpenAI-compatible • your hardware, your rules"
- **Headline**: "Built for Those Who Value Independence" (with gradient on "Independence")
- **Subtitle**: "From solo developers to enterprises, rbee adapts to your needs. Own your AI infrastructure without compromising on power or flexibility."
- **Action Rail**:
  - Primary CTA: "Explore use cases" (links to #use-cases)
  - Secondary CTA: "See the architecture" (links to #architecture)
  - Audience Chips: Developers, Enterprise, Homelab (quick navigation)
- **Quick Proof Row**: "Self-hosted control • OpenAI-compatible API • CUDA • Metal • CPU"
- **Visual**: Homelab desk illustration with laptop, GPU unit, and "your models your rules" note

## When to Use
- As the first section on the Use Cases page (/use-cases)
- To communicate independence and control value proposition
- To provide quick navigation to specific personas

## Content Requirements
- **Headline:** Independence and control focused
- **Audience Chips:** Quick navigation to persona sections
- **Visual:** Represents self-hosted/homelab aesthetic
- **Proof Points:** Technical capabilities (OpenAI-compatible, CUDA, Metal, CPU)

## Marketing Strategy
- **Target Audience:** Broad (developers, enterprises, homelabbers) who value independence
- **Primary Message:** "Own your AI infrastructure without compromising"
- **Emotional Appeal:** Freedom (independence) + Control (your hardware, your rules)
- **CTAs:** 
  - Primary: "Explore use cases" - see specific scenarios
  - Secondary: "See the architecture" - technical deep dive
- **Copy Tone:** Empowering, independence-focused, technical

## Variants
- **Default**: Full hero with all elements
- **ScenarioDriven**: Emphasize scenario storytelling and persona navigation
- **TechnicalFocus**: Lead with OpenAI-compatible API and technical capabilities

## Examples
\`\`\`tsx
import { UseCasesHero } from '@rbee/ui/organisms/UseCases/UseCasesHero'

// Simple usage - no props needed
<UseCasesHero />
\`\`\`

## Used In
- Use Cases page (/use-cases)

## Related Components
- Button (for CTAs)
- Badge (for kicker)
- UseCasesPrimary (linked via #use-cases)
- UseCasesIndustry (linked via #architecture)

## Accessibility
- **Semantic HTML**: Proper heading hierarchy (h1)
- **Keyboard Navigation**: All links and buttons are keyboard accessible
- **Focus States**: Visible focus indicators on interactive elements
- **ARIA Labels**: Image has descriptive alt text
- **Screen Readers**: Proof points are properly announced
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof UseCasesHero>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default UseCasesHero as used on /use-cases page.
 * Shows independence-focused messaging with audience navigation.
 */
export const UseCasesPageDefault: Story = {}

/**
 * Variant emphasizing scenario storytelling.
 * Focuses on persona-driven narratives and quick navigation.
 */
export const ScenarioDriven: Story = {}

/**
 * Variant emphasizing technical capabilities.
 * Leads with OpenAI-compatible API and hardware support.
 */
export const TechnicalFocus: Story = {}
