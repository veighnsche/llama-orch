import type { Meta, StoryObj } from '@storybook/react'
import { DevelopersProblem } from './DevelopersProblem'

const meta = {
	title: 'Organisms/Developers/DevelopersProblem',
	component: DevelopersProblem,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The DevelopersProblem section highlights the specific risks developers face when building complex codebases with AI assistance from external providers. It uses the shared ProblemSection component with developer-specific messaging that emphasizes dependency risk, cost escalation, and codebase maintainability.

## Composition
This organism wraps ProblemSection with:
- **Kicker**: "The Hidden Cost of Dependency"
- **Title**: "The Hidden Risk of AI-Assisted Development"
- **Subtitle**: Developer-specific context about building with AI
- **3 Problem Cards**:
  1. **The Model Changes**: Workflow destruction (destructive tone)
  2. **The Price Increases**: Cost spiral (primary tone, "Cost increase: 10x" tag)
  3. **The Provider Shuts Down**: Codebase becomes unmaintainable (destructive tone)
- **CTA Banner**: "Take Control" + "View Documentation"

## Marketing Strategy

### Target Sub-Audience
**Primary**: Developers using AI coding assistants (Cursor, Zed, Continue, GitHub Copilot)
**Secondary**: Engineering teams building AI-assisted codebases

### Developers Page vs. Home Page Problem Messaging

**Home Page ProblemSection:**
- Problems: "Unpredictable costs, vendor lock-in, privacy concerns"
- Audience: General (all personas)
- Tone: Broad pain points
- Focus: Cost, privacy, control

**Developers Page DevelopersProblem:**
- Problems: "Model changes break workflow, price increases spiral, provider shuts down"
- Audience: Developers specifically
- Tone: Technical dependency risk
- Focus: Workflow stability, codebase maintainability, cost predictability
- **Critical Difference**: Emphasizes "complex codebases built with AI assistance" becoming unmaintainable

### Copy Analysis
- **Technical level**: Intermediate
- **Fear factors**:
  - "Your workflow is destroyed. Your team is blocked." (immediate impact)
  - "$20/month becomes $200/month. Multiply by your team size." (economic fear)
  - "Your complex codebase—built with AI assistance—becomes unmaintainable overnight." (existential threat)
- **Tags**: "High risk", "Cost increase: 10x", "Critical failure" (urgency)

### Conversion Elements
- **Primary CTA**: "Take Control" (action-oriented, empowerment)
- **Secondary CTA**: "View Documentation" (educational path)
- **CTA Copy**: "Heavy, complicated codebases built with AI assistance are a ticking time bomb if you depend on external providers."
  - Uses fear (ticking time bomb) to drive action

## When to Use
- On the Developers page after the hero section
- To establish the problem before presenting the solution
- To create urgency around dependency risk

## Content Requirements
- **Kicker**: Context setter (dependency theme)
- **Title**: Problem statement (risk-focused)
- **Subtitle**: Audience-specific context
- **Problem Cards**: 3 specific risks with icons, tones, and tags
- **CTA Banner**: Strong copy + dual CTAs

## Variants
- **Default**: Full problem section with 3 cards
- **Comparison**: Side-by-side with home page problems
- **Single Problem**: Focus on one risk (model changes)

## Examples
\`\`\`tsx
import { DevelopersProblem } from '@rbee/ui/organisms/Developers/DevelopersProblem'

// Simple usage - no props needed (uses defaults)
<DevelopersProblem />
\`\`\`

## Used In
- Developers page (\`/developers\`)

## Comparison to Home Page

### Home Page ProblemSection:
- **Problems**: 
  1. "The model changes" (general)
  2. "The price increases" (general)
  3. "The provider shuts down" (general)
- **Tone**: Broad, appeals to all personas
- **Focus**: Cost, privacy, vendor lock-in

### Developers Page DevelopersProblem:
- **Problems**:
  1. "The Model Changes" → "Your AI assistant updates overnight. Code generation breaks. Your workflow is destroyed."
  2. "The Price Increases" → "$20/month becomes $200/month. Multiply by your team size."
  3. "The Provider Shuts Down" → "Your complex codebase—built with AI assistance—becomes unmaintainable overnight."
- **Tone**: Technical, developer-specific
- **Focus**: Workflow stability, codebase maintainability, team impact

**Key Difference**: Developers version emphasizes "complex codebases built with AI assistance" as the asset at risk, while home version emphasizes general dependency.

## Technical Implementation
This component is a wrapper around the shared \`ProblemSection\` component, demonstrating the pattern of reusing shared components with page-specific defaults.

## Related Components
- ProblemSection (shared base component)
- Home page ProblemSection variant
- Enterprise page EnterpriseProblem variant

## Accessibility
- **Keyboard Navigation**: All CTAs keyboard accessible
- **ARIA Labels**: Icons marked as decorative with aria-hidden
- **Semantic HTML**: Uses \`<section>\` with proper heading hierarchy
- **Color Contrast**: Tone-based colors meet WCAG AA standards
- **Focus States**: Visible focus indicators on CTAs
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof DevelopersProblem>

export default meta
type Story = StoryObj<typeof meta>

export const DevelopersPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story: `Default problem section for the Developers page with exact copy from \`/developers\`. Shows 3 developer-specific risks: model changes, price increases, and provider shutdown. Note the emphasis on "complex codebases built with AI assistance" as the asset at risk.

**Problem 1: The Model Changes**
- **Icon**: AlertTriangle (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "Your AI assistant updates overnight. Suddenly, code generation breaks. Your workflow is destroyed. Your team is blocked."
- **Tag**: "High risk"
- **Target**: Developers using AI coding assistants (Cursor, Zed, Continue, GitHub Copilot)
- **Why this pain point**: This is the #1 fear for developers building with AI assistance. When Claude/GPT/Copilot updates, code generation patterns change. What worked yesterday breaks today. Your team's velocity drops to zero. This is a visceral, immediate pain that developers have experienced firsthand. The copywriter chose this because it's the most relatable pain point—every developer using AI has experienced a breaking change.

**Problem 2: The Price Increases**
- **Icon**: DollarSign (blue/primary)
- **Tone**: Primary (important, cost-focused)
- **Copy**: "$20/month becomes $200/month. Multiply by your team size. Your AI infrastructure costs spiral out of control."
- **Tag**: "Cost increase: 10x"
- **Target**: Engineering managers, team leads, CTOs
- **Why this pain point**: AI tooling costs are unpredictable and rising. GitHub Copilot went from $10/mo to $19-39/mo. Cursor is $20/mo per seat. For a 10-person team, that's $200-400/month. The copywriter chose "10x" because it's a realistic multiplier that's happened in the AI tooling market. This creates budget anxiety and makes AI tooling a line-item risk. The emphasis on "multiply by your team size" makes the pain concrete—it's not $20, it's $200-2000 depending on team size.

**Problem 3: The Provider Shuts Down**
- **Icon**: Lock (red/destructive)
- **Tone**: Destructive (critical problem)
- **Copy**: "API deprecated. Service discontinued. Your complex codebase—built with AI assistance—becomes unmaintainable overnight."
- **Tag**: "Critical failure"
- **Target**: Developers building long-term codebases with AI assistance
- **Why this pain point**: This addresses the existential risk of dependency. When you build complex codebases with AI assistance, you're creating technical debt that's tied to a specific provider. If that provider shuts down or changes their API, your codebase becomes unmaintainable. The copywriter chose "complex codebase—built with AI assistance" to emphasize the asset at risk. This is the "vendor lock-in" fear taken to its logical extreme. The short, punchy sentences ("API deprecated. Service discontinued.") create urgency and finality.`,
			},
		},
	},
}

export const ComparisonToHomePage: Story = {
	render: () => (
		<div className="space-y-8">
			<div className="bg-primary/10 p-6 text-center">
				<h3 className="text-xl font-bold">Developers Page Problem (Below)</h3>
				<p className="text-muted-foreground">
					Focus: Workflow destruction, codebase maintainability, team impact
				</p>
			</div>
			<DevelopersProblem />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Messaging Comparison</h3>
				<div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
					<div>
						<h4 className="font-semibold mb-3">Home Page Problems</h4>
						<div className="space-y-3 text-sm">
							<div>
								<strong>Problem 1:</strong> "The model changes"
								<br />
								<span className="text-muted-foreground">
									→ General: "Code generation breaks; workflows stall"
								</span>
							</div>
							<div>
								<strong>Problem 2:</strong> "The price increases"
								<br />
								<span className="text-muted-foreground">
									→ General: "$20/month becomes $200/month—multiplied by your team"
								</span>
							</div>
							<div>
								<strong>Problem 3:</strong> "The provider shuts down"
								<br />
								<span className="text-muted-foreground">
									→ General: "APIs get deprecated. Your AI-built code becomes unmaintainable"
								</span>
							</div>
							<div className="pt-2">
								<strong>Tone:</strong> Broad appeal, general audience
							</div>
						</div>
					</div>
					<div>
						<h4 className="font-semibold mb-3">Developers Page Problems</h4>
						<div className="space-y-3 text-sm">
							<div>
								<strong>Problem 1:</strong> "The Model Changes"
								<br />
								<span className="text-muted-foreground">
									→ Developer-specific: "Your workflow is destroyed. Your team is blocked."
								</span>
							</div>
							<div>
								<strong>Problem 2:</strong> "The Price Increases"
								<br />
								<span className="text-muted-foreground">
									→ Developer-specific: "Multiply by your team size. Your AI infrastructure costs spiral."
								</span>
							</div>
							<div>
								<strong>Problem 3:</strong> "The Provider Shuts Down"
								<br />
								<span className="text-muted-foreground">
									→ Developer-specific: "Your complex codebase—built with AI assistance—becomes unmaintainable
									overnight."
								</span>
							</div>
							<div className="pt-2">
								<strong>Tone:</strong> Technical, emphasizes codebase as asset at risk
							</div>
						</div>
					</div>
				</div>
				<div className="mt-6 p-4 bg-background rounded-lg max-w-3xl mx-auto">
					<strong>Key Difference:</strong> Developers version emphasizes "complex codebases built with AI assistance"
					as the critical asset that becomes unmaintainable. Home version focuses on general dependency and cost.
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Side-by-side comparison of Developers problem messaging vs. Home page. Developers version is more technical and emphasizes workflow destruction and codebase maintainability.',
			},
		},
	},
}

export const SingleProblemFocus: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersProblem />
			<div className="bg-muted p-8 text-center">
				<h3 className="text-xl font-bold mb-4">Alternative: Single Problem Deep Dive</h3>
				<p className="text-muted-foreground max-w-2xl mx-auto">
					For landing pages or ads, consider focusing on just one problem (e.g., "The Model Changes") with more
					detail and a stronger CTA. This creates a clearer narrative and stronger conversion path.
				</p>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Alternative approach: Focus on a single problem (e.g., model changes breaking workflows) for clearer messaging in ads or landing pages.',
			},
		},
	},
}
