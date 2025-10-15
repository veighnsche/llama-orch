import type { Meta, StoryObj } from '@storybook/react'
import { DevelopersFeatures } from './DevelopersFeatures'

const meta = {
	title: 'Organisms/Developers/DevelopersFeatures',
	component: DevelopersFeatures,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The DevelopersFeatures section showcases developer-focused features like API compatibility, SDKs, debugging tools, and integration capabilities. It emphasizes technical features that matter to developers building with AI.

## Composition
This organism contains:
- **Section Title**: "Features for Developers"
- **Feature Grid**: Developer-specific features
- **Technical Details**: API compatibility, SDK support, debugging, monitoring
- **Integration Examples**: How it works with developer tools

## Marketing Strategy

### Target Sub-Audience
**Primary**: Developers evaluating technical capabilities
**Secondary**: Engineering teams assessing integration complexity

### Page-Specific Messaging
- **Developers page**: Technical features (API, SDKs, debugging, monitoring)
- **Technical level**: Intermediate to Advanced
- **Focus**: Developer experience and tooling

### Copy Analysis
- **Technical level**: Intermediate to Advanced
- **Key features**:
  - OpenAI-compatible API
  - SDK support (TypeScript, Python)
  - Debugging and monitoring tools
  - Integration with IDEs (Zed, Cursor, Continue)
- **Proof points**: Shows comprehensive developer tooling

### Conversion Elements
- **Technical credibility**: Shows the platform is developer-friendly
- **Reduces friction**: "Works with your existing tools"
- **Educational**: Helps developers understand capabilities

## When to Use
- On the Developers page after "How It Works"
- To showcase technical capabilities
- To demonstrate developer experience quality

## Content Requirements
- **Title**: Clear section heading
- **Features**: Developer-specific technical features
- **Details**: API docs, SDK examples, tool integrations
- **Visual aids**: Screenshots or code examples

## Examples
\`\`\`tsx
import { DevelopersFeatures } from '@rbee/ui/organisms/Developers/DevelopersFeatures'

<DevelopersFeatures />
\`\`\`

## Used In
- Developers page (\`/developers\`)

## Related Components
- FeaturesSection (home page)
- CoreFeaturesTabs (features page)

## Accessibility
- **Keyboard Navigation**: All interactive elements keyboard accessible
- **ARIA Labels**: Proper labels on feature cards
- **Semantic HTML**: Uses proper heading hierarchy
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof DevelopersFeatures>

export default meta
type Story = StoryObj<typeof meta>

export const DevelopersPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default features section for the Developers page. Shows developer-specific features like API compatibility, SDKs, debugging tools, and IDE integrations.',
			},
		},
	},
}

export const CoreFeaturesOnly: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersFeatures />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Alternative: Core Features Only</h3>
				<div className="max-w-2xl mx-auto">
					<p className="text-sm text-muted-foreground mb-4">
						For landing pages or quick overviews, consider showing only the 3-4 most important features:
					</p>
					<div className="space-y-2 text-sm">
						<div className="bg-background p-3 rounded-lg">
							<strong>1. OpenAI-Compatible API</strong> - Drop-in replacement
						</div>
						<div className="bg-background p-3 rounded-lg">
							<strong>2. TypeScript SDK</strong> - First-class TypeScript support
						</div>
						<div className="bg-background p-3 rounded-lg">
							<strong>3. IDE Integration</strong> - Works with Zed, Cursor, Continue
						</div>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Alternative minimal version showing only the 3-4 most important developer features.',
			},
		},
	},
}

export const TechnicalDepthComparison: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersFeatures />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Developer Features vs. General Features</h3>
				<div className="max-w-3xl mx-auto grid md:grid-cols-2 gap-4 text-sm">
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Home Page Features:</strong>
						<ul className="space-y-1 text-muted-foreground">
							<li>• Multi-GPU orchestration</li>
							<li>• Zero ongoing costs</li>
							<li>• Complete privacy</li>
							<li>• Easy setup</li>
						</ul>
						<p className="text-xs text-muted-foreground mt-2">→ Broad appeal, benefit-focused</p>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Developers Page Features:</strong>
						<ul className="space-y-1 text-muted-foreground">
							<li>• OpenAI-compatible API</li>
							<li>• TypeScript/Python SDKs</li>
							<li>• IDE integrations</li>
							<li>• Debugging & monitoring</li>
						</ul>
						<p className="text-xs text-muted-foreground mt-2">→ Developer-specific, tooling-focused</p>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Comparison of feature messaging: Developers page focuses on technical tooling, home page focuses on benefits.',
			},
		},
	},
}
