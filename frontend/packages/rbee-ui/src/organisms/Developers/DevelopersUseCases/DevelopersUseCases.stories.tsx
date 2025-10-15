import type { Meta, StoryObj } from '@storybook/react'
import { DevelopersUseCases } from './DevelopersUseCases'

const meta = {
	title: 'Organisms/Developers/DevelopersUseCases',
	component: DevelopersUseCases,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The DevelopersUseCases section showcases specific developer use cases for rbee, such as AI-assisted coding, code review, documentation generation, and test generation. It demonstrates practical applications for developers.

## Composition
This organism contains:
- **Section Title**: "Use Cases for Developers"
- **Use Case Cards**: Specific developer scenarios
- **Technical Examples**: Code snippets or workflow descriptions
- **Persona Mapping**: Different developer types and their use cases

## Marketing Strategy

### Target Sub-Audience
**Primary**: Developers looking for specific use cases
**Secondary**: Engineering teams evaluating fit for their workflow

### Page-Specific Messaging
- **Developers page**: Developer-specific use cases (coding, review, docs, tests)
- **Technical level**: Intermediate
- **Focus**: Practical applications in developer workflow

### Copy Analysis
- **Technical level**: Intermediate
- **Use cases**:
  - AI-assisted coding (Cursor, Zed integration)
  - Code review automation
  - Documentation generation
  - Test generation
  - Refactoring assistance
- **Persona differences**: Solo developer vs. team vs. enterprise

### Conversion Elements
- **Relatability**: "This is how you'd actually use it"
- **Specificity**: Concrete examples reduce uncertainty
- **Persona matching**: Helps developers see themselves in the use case

## When to Use
- On the Developers page after features section
- To show practical applications
- To help developers envision using rbee

## Content Requirements
- **Title**: Clear section heading
- **Use Cases**: 3-5 specific developer scenarios
- **Details**: Workflow descriptions or code examples
- **Personas**: Different developer types

## Examples
\`\`\`tsx
import { DevelopersUseCases } from '@rbee/ui/organisms/Developers/DevelopersUseCases'

<DevelopersUseCases />
\`\`\`

## Used In
- Developers page (\`/developers\`)

## Comparison to Home Page

### Home Page UseCasesSection:
- **Use Cases**: Solo developer, small team, homelab enthusiast, enterprise
- **Focus**: Broad personas with economic benefits
- **Tone**: Accessible to all

### Developers Page DevelopersUseCases:
- **Use Cases**: AI coding, code review, docs generation, test generation
- **Focus**: Specific developer workflows
- **Tone**: Technical, workflow-focused

**Key Difference**: Home page shows *who* uses rbee (personas), Developers page shows *how* developers use rbee (workflows).

## Related Components
- UseCasesSection (home page)
- Use Cases page organisms

## Accessibility
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **ARIA Labels**: Proper labels on use case cards
- **Semantic HTML**: Uses proper heading hierarchy
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof DevelopersUseCases>

export default meta
type Story = StoryObj<typeof meta>

export const DevelopersPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default use cases section for the Developers page. Shows developer-specific workflows like AI coding, code review, documentation generation, and test generation.',
			},
		},
	},
}

export const SingleUseCaseDeepDive: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersUseCases />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Alternative: Single Use Case Deep Dive</h3>
				<div className="max-w-2xl mx-auto">
					<p className="text-sm text-muted-foreground mb-4">
						For landing pages targeting a specific use case (e.g., "AI Coding Assistant"), consider showing just
						one use case with more detail, code examples, and a stronger CTA.
					</p>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Example: AI Coding Assistant Deep Dive</strong>
						<ul className="space-y-2 text-sm text-muted-foreground">
							<li>• Show actual code generation examples</li>
							<li>• Compare to Cursor/Copilot workflow</li>
							<li>• Emphasize cost savings ($20/mo → $0)</li>
							<li>• Show IDE integration screenshots</li>
							<li>• Strong CTA: "Start coding with AI for free"</li>
						</ul>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Alternative approach: Deep dive on a single use case (e.g., AI coding) for targeted landing pages or campaigns.',
			},
		},
	},
}

export const ComparisonToHomePage: Story = {
	render: () => (
		<div className="space-y-8">
			<div className="bg-primary/10 p-6 text-center">
				<h3 className="text-xl font-bold">Developers Page Use Cases (Below)</h3>
				<p className="text-muted-foreground">Focus: Developer workflows (HOW developers use rbee)</p>
			</div>
			<DevelopersUseCases />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Use Case Messaging Comparison</h3>
				<div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-6 text-sm">
					<div>
						<h4 className="font-semibold mb-3">Home Page Use Cases</h4>
						<div className="space-y-2">
							<div className="bg-background p-3 rounded-lg">
								<strong>Solo Developer</strong>
								<br />
								<span className="text-muted-foreground">→ Persona: "$0/month AI costs"</span>
							</div>
							<div className="bg-background p-3 rounded-lg">
								<strong>Small Team</strong>
								<br />
								<span className="text-muted-foreground">→ Persona: "$6,000+ saved per year"</span>
							</div>
							<div className="bg-background p-3 rounded-lg">
								<strong>Homelab Enthusiast</strong>
								<br />
								<span className="text-muted-foreground">→ Persona: "Idle GPUs → productive"</span>
							</div>
						</div>
						<p className="text-xs text-muted-foreground mt-3">→ Focus: WHO uses rbee (personas + economics)</p>
					</div>
					<div>
						<h4 className="font-semibold mb-3">Developers Page Use Cases</h4>
						<div className="space-y-2">
							<div className="bg-background p-3 rounded-lg">
								<strong>AI-Assisted Coding</strong>
								<br />
								<span className="text-muted-foreground">→ Workflow: Code generation in IDE</span>
							</div>
							<div className="bg-background p-3 rounded-lg">
								<strong>Code Review</strong>
								<br />
								<span className="text-muted-foreground">→ Workflow: Automated review suggestions</span>
							</div>
							<div className="bg-background p-3 rounded-lg">
								<strong>Documentation</strong>
								<br />
								<span className="text-muted-foreground">→ Workflow: Auto-generate docs from code</span>
							</div>
						</div>
						<p className="text-xs text-muted-foreground mt-3">→ Focus: HOW developers use rbee (workflows)</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Comparison: Home page shows WHO uses rbee (personas), Developers page shows HOW developers use rbee (workflows).',
			},
		},
	},
}
