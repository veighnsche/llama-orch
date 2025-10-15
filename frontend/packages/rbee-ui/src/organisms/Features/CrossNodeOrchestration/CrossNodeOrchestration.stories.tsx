import type { Meta, StoryObj } from '@storybook/react'
import { CrossNodeOrchestration } from './CrossNodeOrchestration'

const meta = {
	title: 'Organisms/Features/CrossNodeOrchestration',
	component: CrossNodeOrchestration,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The CrossNodeOrchestration section explains rbee's distributed execution capabilities—how it orchestrates AI workloads across multiple machines in a home network. It covers pool registry management (adding remote machines) and automatic worker provisioning (spawning workers on demand via SSH).

## Composition
This organism contains:
- **Section Title**: "Cross-Pool Orchestration"
- **Section Subtitle**: "Seamlessly orchestrate AI workloads across your entire network"
- **Badge**: "Distributed execution"
- **Two Cards**:
  1. **Pool Registry Management**: CLI example showing how to add remote machines
  2. **Automatic Worker Provisioning**: Diagram showing queen-rbee → rbee-hive → worker-rbee flow

## Marketing Strategy

### Target Sub-Audience
**Primary**: Technical users with multiple machines (homelab enthusiasts, small teams)
**Secondary**: Enterprise evaluators assessing distributed capabilities

### Page-Specific Messaging
- **Features page**: Technical deep dive into distributed orchestration
- **Technical level**: Advanced
- **Focus**: How cross-node orchestration actually works

### Copy Analysis
- **Technical level**: Advanced
- **Card 1 - Pool Registry Management**:
  - Benefit: "Configure remote machines once"
  - Proof: CLI example showing \`rbee-keeper setup add-node\` with SSH parameters
  - Features: Automatic detection, SSH validation, zero config on remote nodes
- **Card 2 - Automatic Worker Provisioning**:
  - Benefit: "rbee spawns workers via SSH on demand and shuts them down cleanly"
  - Proof: Architecture diagram (queen-rbee → SSH → rbee-hive → Spawns → worker-rbee)
  - Features: On-demand start, clean shutdown, no daemon drift

### Conversion Elements
- **Technical credibility**: Shows the system is well-architected
- **Simplicity**: "Configure once, use everywhere"
- **Reliability**: "Clean shutdown, no daemon drift"

## When to Use
- On the Features page after CoreFeaturesTabs
- To explain distributed execution capabilities
- To show technical architecture
- To demonstrate homelab-scale orchestration

## Content Requirements
- **Section Title**: Clear heading
- **Section Subtitle**: Brief overview
- **Badge**: Feature category
- **CLI Example**: Actual commands for adding nodes
- **Architecture Diagram**: Visual showing component relationships
- **Feature Strips**: Key benefits (3 per card)

## Variants
- **Default**: Full section with both cards
- **Simplified Explanation**: Less technical, more conceptual
- **With Diagram**: Emphasis on visual architecture

## Examples
\`\`\`tsx
import { CrossNodeOrchestration } from '@rbee/ui/organisms/Features/CrossNodeOrchestration'

<CrossNodeOrchestration />
\`\`\`

## Used In
- Features page (\`/features\`)

## Technical Implementation
- Uses SectionContainer for consistent layout
- IconBox for feature icons
- Responsive grid (lg:grid-cols-2)
- Animated cards (fade-in slide-in)

## Related Components
- SectionContainer
- IconBox
- Badge

## Accessibility
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **ARIA Labels**: CLI example has aria-label="Pool registry CLI example"
- **Semantic HTML**: Uses proper heading hierarchy
- **Focus States**: Visible focus indicators
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof CrossNodeOrchestration>

export default meta
type Story = StoryObj<typeof meta>

export const FeaturesPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default cross-node orchestration section for the Features page. Shows pool registry management (CLI for adding remote machines) and automatic worker provisioning (architecture diagram). Demonstrates distributed execution capabilities.',
			},
		},
	},
}

export const SimplifiedExplanation: Story = {
	render: () => (
		<div className="space-y-8">
			<CrossNodeOrchestration />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Alternative: Simplified Explanation</h3>
				<div className="max-w-2xl mx-auto">
					<p className="text-sm text-muted-foreground mb-4">
						For less technical audiences, consider simplifying the explanation:
					</p>
					<div className="bg-background p-4 rounded-lg space-y-3 text-sm">
						<div>
							<strong>Step 1:</strong> Tell rbee about your other machines
							<br />
							<span className="text-muted-foreground">→ One command per machine</span>
						</div>
						<div>
							<strong>Step 2:</strong> Run inference on any machine
							<br />
							<span className="text-muted-foreground">→ rbee handles the rest (SSH, spawning, cleanup)</span>
						</div>
						<div>
							<strong>Result:</strong> All your machines work together
							<br />
							<span className="text-muted-foreground">→ No manual setup on remote machines</span>
						</div>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Alternative simplified explanation for less technical audiences. Reduces complexity while maintaining key benefits.',
			},
		},
	},
}

export const WithDiagramEmphasis: Story = {
	render: () => (
		<div className="space-y-8">
			<div className="bg-primary/10 p-6 text-center">
				<h3 className="text-xl font-bold">Architecture Diagram Emphasis</h3>
				<p className="text-muted-foreground">
					The diagram shows how queen-rbee orchestrates workers across machines via SSH
				</p>
			</div>
			<CrossNodeOrchestration />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Diagram Strategy</h3>
				<div className="max-w-3xl mx-auto space-y-4 text-sm">
					<p className="text-muted-foreground">
						The architecture diagram is a key differentiator. It shows:
					</p>
					<div className="grid md:grid-cols-2 gap-4">
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Technical Credibility</strong>
							<p className="text-muted-foreground">
								Shows the system is well-architected with clear component separation (queen-rbee, rbee-hive, worker-rbee)
							</p>
						</div>
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Simplicity</strong>
							<p className="text-muted-foreground">
								Despite being distributed, the architecture is simple: orchestrator → pool manager → worker
							</p>
						</div>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Key Differentiator:</strong>
						<p className="text-muted-foreground">
							"No daemon drift" and "Clean shutdown" are unique selling points. Most distributed systems leave orphaned processes. rbee doesn't.
						</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Emphasis on the architecture diagram as a key differentiator. Shows technical credibility and simplicity despite distributed nature.',
			},
		},
	},
}
