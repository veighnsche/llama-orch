import type { Meta, StoryObj } from '@storybook/react'
import { DevelopersSolution } from './DevelopersSolution'

const meta = {
	title: 'Organisms/Developers/DevelopersSolution',
	component: DevelopersSolution,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The DevelopersSolution section presents the rbee solution specifically for developers. It wraps the shared SolutionSection component with developer-focused messaging that emphasizes hardware ownership, OpenAI compatibility, and the 4-step getting started process.

## Composition
This organism wraps SolutionSection with:
- **Kicker**: "How rbee Works"
- **Title**: "Your Hardware. Your Models. Your Control."
- **Subtitle**: Technical explanation of orchestration
- **4 Feature Cards**:
  1. Zero Ongoing Costs (DollarSign icon)
  2. Complete Privacy (Lock icon)
  3. You Decide When to Update (Zap icon)
  4. Use All Your Hardware (Cpu icon)
- **4 Steps**:
  1. Install rbee
  2. Add Your Hardware
  3. Download Models
  4. Start Building
- **Dual CTAs**: "Get Started" + "View Documentation"

## Marketing Strategy

### Target Sub-Audience
**Primary**: Developers ready to take action after seeing the problem
**Secondary**: Developers evaluating alternatives to cloud AI providers

### Page-Specific Messaging
- **Developers page**: Emphasizes "Your Hardware. Your Models. Your Control." (ownership)
- **Technical depth**: Intermediate - explains orchestration, backends (CUDA, Metal, CPU)
- **Action-oriented**: 4-step process shows it's achievable

### Copy Analysis
- **Technical level**: Intermediate
- **Key benefits**:
  - "Zero Ongoing Costs" → "Pay only for electricity. No subscriptions or per-token fees."
  - "Complete Privacy" → "Code never leaves your network. GDPR-friendly by default."
  - "You Decide When to Update" → "Models change only when you choose—no surprise breakages."
  - "Use All Your Hardware" → "Orchestrate CUDA, Metal, and CPU. Every chip contributes."
- **Process clarity**: 4 clear steps from install to building
- **Proof points**: OpenAI-compatible API, auto-detection, Hugging Face integration

### Conversion Elements
- **Primary CTA**: "Get Started" (action)
- **Secondary CTA**: "View Documentation" (education)
- **Steps**: Show it's achievable (reduces friction)

## When to Use
- On the Developers page after the problem section
- To present the solution and getting started process
- To show technical capabilities without overwhelming

## Content Requirements
- **Kicker**: Context setter
- **Title**: Benefit-focused headline
- **Subtitle**: Technical explanation
- **Feature Cards**: 4 key benefits with icons
- **Steps**: 4-step process
- **CTAs**: Dual action paths

## Variants
- **Default**: Full solution with 4 features and 4 steps
- **Features Only**: Just the benefits without steps
- **Steps Only**: Just the process without benefits

## Examples
\`\`\`tsx
import { DevelopersSolution } from '@rbee/ui/organisms/Developers/DevelopersSolution'

// Simple usage - no props needed
<DevelopersSolution />
\`\`\`

## Used In
- Developers page (\`/developers\`)

## Technical Implementation
This component wraps the shared \`SolutionSection\` component with developer-specific defaults, demonstrating the pattern of reusing shared components across pages.

## Related Components
- SolutionSection (shared base component)
- Home page SolutionSection variant

## Accessibility
- **Keyboard Navigation**: All CTAs keyboard accessible
- **ARIA Labels**: Icons marked as decorative with aria-hidden
- **Semantic HTML**: Uses \`<section>\` with proper heading hierarchy
- **Focus States**: Visible focus indicators on CTAs
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof DevelopersSolution>

export default meta
type Story = StoryObj<typeof meta>

export const DevelopersPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default solution section for the Developers page. Shows 4 key benefits (cost, privacy, control, hardware) and 4-step getting started process. Emphasizes ownership: "Your Hardware. Your Models. Your Control."',
			},
		},
	},
}

export const AlternativeBenefits: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersSolution />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Alternative Benefit Emphasis</h3>
				<div className="max-w-3xl mx-auto space-y-4 text-sm">
					<div>
						<strong>Current Order:</strong> Cost → Privacy → Control → Hardware
						<br />
						<span className="text-muted-foreground">
							→ Leads with economic benefit (appeals to budget-conscious developers)
						</span>
					</div>
					<div>
						<strong>Alternative A:</strong> Privacy → Control → Cost → Hardware
						<br />
						<span className="text-muted-foreground">
							→ Leads with privacy (appeals to security-conscious developers)
						</span>
					</div>
					<div>
						<strong>Alternative B:</strong> Control → Hardware → Cost → Privacy
						<br />
						<span className="text-muted-foreground">
							→ Leads with control (appeals to autonomy-focused developers)
						</span>
					</div>
					<div className="pt-2">
						<strong>Testing Recommendation:</strong> A/B test the order. Cost-first likely converts best for
						indie developers, privacy-first for enterprise developers.
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Alternative benefit ordering for A/B testing. Current order leads with cost (economic benefit), but privacy-first or control-first may convert better for specific developer segments.',
			},
		},
	},
}

export const SimplifiedSteps: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersSolution />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Alternative: 2-Step Simplified Process</h3>
				<div className="max-w-2xl mx-auto space-y-4">
					<div className="bg-background p-4 rounded-lg">
						<strong>Step 1:</strong> Install rbee
						<br />
						<span className="text-sm text-muted-foreground">
							One command. Auto-detects all hardware. Downloads models.
						</span>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong>Step 2:</strong> Point your code to localhost
						<br />
						<span className="text-sm text-muted-foreground">
							Change OPENAI_API_BASE. Everything else stays the same.
						</span>
					</div>
					<p className="text-sm text-muted-foreground text-center pt-2">
						For landing pages or ads, a 2-step process may feel more achievable and reduce friction.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Alternative simplified 2-step process for landing pages or ads. Reduces perceived complexity and friction.',
			},
		},
	},
}
