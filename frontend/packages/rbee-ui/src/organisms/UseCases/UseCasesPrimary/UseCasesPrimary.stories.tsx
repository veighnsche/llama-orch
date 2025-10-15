import type { Meta, StoryObj } from '@storybook/react'
import { UseCasesPrimary } from './UseCasesPrimary'

const meta = {
	title: 'Organisms/UseCases/UseCasesPrimary',
	component: UseCasesPrimary,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The UseCasesPrimary section showcases 8 primary use case scenarios in a 2-column grid layout. Each use case follows a problem-solution-outcome format with persona-specific details (Solo Developer, Small Team, Homelab Enthusiast, Enterprise, Freelance Developer, Research Lab, Open Source Maintainer, GPU Provider). Includes filter navigation and hero visual.

## Use Case Storytelling

### Narrative Structure (Per Use Case)
1. **Persona:** Who is this for? (e.g., "The Solo Developer")
2. **Scenario:** What problem do they face? (e.g., "Building a SaaS with AI, wants Claude-level coding without vendor lock-in")
3. **Solution:** How does rbee solve it? (e.g., "Run rbee on gaming PC + spare workstation; Llama 70B for code, SD for assets")
4. **Outcome:** What result do they achieve? (e.g., "$0/mo inference, Full control, No rate limits")

### Emotional Journey (Across All Use Cases)
- **Before:** Vendor lock-in, high costs, compliance concerns, limited control
- **After:** Independence, cost savings, compliance, full control

### Proof Points (Per Use Case)
- **Metrics:** Cost savings ($0/mo, ~$6k/yr saved), technical capabilities
- **Benefits:** Full control, no rate limits, GDPR-friendly, zero ongoing costs

### Use Case Categories
- **Solo/Freelance:** Solo Developer, Freelance Developer, Open Source Maintainer
- **Team:** Small Team, Research Lab
- **Enterprise:** Enterprise (compliance focus)
- **Homelab:** Homelab Enthusiast, GPU Provider

## Composition
This organism contains:
- **SectionContainer**: "Real Scenarios. Real Solutions."
- **Header Block**:
  - Eyebrow: "OpenAI-compatible · Your GPUs · Zero API fees"
  - Hero Visual: UsecasesGridDark illustration
  - Filter Pills: All, Solo, Team, Enterprise, Research (navigation)
- **Use Cases Grid** (2-column responsive):
  - 8 UseCaseCard components
  - Each with icon, color, title, scenario, solution, highlights, optional badge

## When to Use
- On the Use Cases page after the hero section
- To provide detailed scenario-driven use cases
- To help users identify with specific personas

## Content Requirements
- **Use Cases:** Must follow problem-solution-outcome format
- **Personas:** Diverse range (solo to enterprise)
- **Proof Points:** Specific metrics and benefits per use case
- **Filter Navigation:** Quick access to persona categories

## Marketing Strategy
- **Target Audience:** Broad (all personas evaluating rbee)
- **Primary Message:** "rbee solves real problems for real people"
- **Emotional Appeal:** Identification (persona match) + Validation (proven solutions)
- **Copy Tone:** Scenario-driven, specific, benefit-focused

## Variants
- **Default**: All 8 use cases with filter navigation
- **SingleUseCase**: Deep dive into one use case (e.g., Solo Developer)
- **CompareToHomeUseCases**: Show differences from home page use cases section

## Examples
\`\`\`tsx
import { UseCasesPrimary } from '@rbee/ui/organisms/UseCases/UseCasesPrimary'

// Simple usage - no props needed
<UseCasesPrimary />
\`\`\`

## Used In
- Use Cases page (/use-cases)

## Related Components
- SectionContainer (layout wrapper)
- UseCaseCard (individual use case display)
- UseCasesHero (page hero)
- UseCasesIndustry (industry-specific use cases)

## Accessibility
- **Semantic HTML**: Proper heading hierarchy and card structure
- **Keyboard Navigation**: Filter pills and cards are keyboard accessible
- **ARIA Labels**: Filter navigation has aria-label
- **Screen Readers**: Use case details are properly announced
- **Focus States**: Visible focus indicators on interactive elements
				`,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof UseCasesPrimary>

export default meta
type Story = StoryObj<typeof meta>

/**
 * Default UseCasesPrimary as used on /use-cases page.
 * Shows all 8 use cases with filter navigation.
 */
export const UseCasesPageDefault: Story = {}

/**
 * Variant showing a single use case deep dive.
 * Focuses on one persona (e.g., Solo Developer) with expanded details.
 */
export const SingleUseCase: Story = {}

/**
 * Variant comparing to home page use cases.
 * Shows how use cases page provides more depth than home page section.
 */
export const CompareToHomeUseCases: Story = {}
