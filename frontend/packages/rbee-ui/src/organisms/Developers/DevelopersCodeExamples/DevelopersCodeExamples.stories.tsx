import type { Meta, StoryObj } from '@storybook/react'
import { DevelopersCodeExamples } from './DevelopersCodeExamples'

const meta = {
	title: 'Organisms/Developers/DevelopersCodeExamples',
	component: DevelopersCodeExamples,
	parameters: {
		layout: 'fullscreen',
		docs: {
			description: {
				component: `
## Overview
The DevelopersCodeExamples section showcases TypeScript code examples using the \`@llama-orch/utils\` SDK. It demonstrates practical code patterns for developers building AI agents and LLM pipelines. This section treats code examples as marketing tools, showing developers exactly how to use rbee.

## Composition
This organism wraps CodeExamplesSection with:
- **Title**: "Build AI agents with llama-orch-utils"
- **Subtitle**: "TypeScript utilities for LLM pipelines and agentic workflows."
- **Footer Note**: "Works with any OpenAI-compatible client."
- **3 Code Examples**:
  1. **Simple code generation**: Basic invoke() usage
  2. **File operations**: Read schema → generate API → write file
  3. **Multi-step agent**: Threaded review + suggestion extraction

## Marketing Strategy

### Target Sub-Audience
**Primary**: TypeScript/JavaScript developers building with AI
**Secondary**: Developers evaluating SDK quality and ease of use

### Code Examples as Marketing Tools

**CRITICAL INSIGHT**: Code examples are not just documentation—they are marketing materials. Each example demonstrates:
1. **Simplicity**: How easy it is to use rbee
2. **Power**: What you can build with it
3. **Quality**: The SDK is well-designed
4. **Credibility**: This is production-ready

### Copy Analysis
- **Technical level**: Intermediate to Advanced
- **Languages**: TypeScript (primary developer audience)
- **Complexity progression**: Simple → File ops → Multi-step agent
- **Code comments**: Minimal (code is self-documenting)
- **Proof points**:
  - Clean API design (\`invoke()\`, \`FileReader\`, \`Thread\`)
  - Composable utilities (\`extractCode\`, \`toMessages\`)
  - Real-world patterns (file operations, threaded conversations)

### Conversion Elements
- **Simplicity**: "This is easy to use"
- **Power**: "I can build complex agents with this"
- **Trust**: "The SDK is well-designed"
- **Action**: Developers want to try it immediately

## When to Use
- On the Developers page after use cases
- To show actual code patterns
- To demonstrate SDK quality
- To inspire developers with what's possible

## Content Requirements
- **Title**: Clear section heading
- **Subtitle**: Context about the SDK
- **Code Examples**: 3-5 examples showing progression
- **Footer Note**: Additional context or compatibility info

## Code Example Strategy

### Example 1: Simple
- **Goal**: Show how easy it is to get started
- **Pattern**: Single function call
- **Message**: "You can do this in 5 lines"

### Example 2: File Operations
- **Goal**: Show practical workflow
- **Pattern**: Read → Process → Write
- **Message**: "This handles real developer tasks"

### Example 3: Multi-step Agent
- **Goal**: Show advanced capabilities
- **Pattern**: Threaded conversation + extraction
- **Message**: "You can build complex agents"

## Examples
\`\`\`tsx
import { DevelopersCodeExamples } from '@rbee/ui/organisms/Developers/DevelopersCodeExamples'

// Simple usage - no props needed
<DevelopersCodeExamples />
\`\`\`

## Used In
- Developers page (\`/developers\`)

## Marketing Documentation

### Why TypeScript?
- **Primary developer audience**: TypeScript is the dominant language for modern web development
- **Type safety**: Shows the SDK has good TypeScript support (important for developers)
- **Ecosystem fit**: Works with existing TypeScript tooling

### Why These Examples?
1. **Simple code generation**: Most common use case (AI coding assistants)
2. **File operations**: Shows practical workflow automation
3. **Multi-step agent**: Shows advanced capabilities (differentiator)

### Code Comment Strategy
- **Minimal comments**: Code should be self-documenting
- **Inline comments**: Only where necessary for clarity
- **No tutorial comments**: This is marketing, not a tutorial

## Related Components
- CodeExamplesSection (shared base component)

## Accessibility
- **Keyboard Navigation**: Code blocks are scrollable and keyboard accessible
- **ARIA Labels**: Code blocks have proper language labels
- **Semantic HTML**: Uses \`<pre>\` and \`<code>\` elements
- **Color Contrast**: Syntax highlighting meets WCAG AA standards
        `,
			},
		},
	},
	tags: ['autodocs'],
} satisfies Meta<typeof DevelopersCodeExamples>

export default meta
type Story = StoryObj<typeof meta>

export const DevelopersPageDefault: Story = {
	parameters: {
		docs: {
			description: {
				story:
					'Default code examples section for the Developers page. Shows 3 TypeScript examples: simple code generation, file operations, and multi-step agent. Code examples are marketing tools that demonstrate simplicity, power, and SDK quality.',
			},
		},
	},
}

export const TypeScriptOnly: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersCodeExamples />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Why TypeScript Only?</h3>
				<div className="max-w-2xl mx-auto space-y-4 text-sm">
					<div className="bg-background p-4 rounded-lg">
						<strong>Strategic Decision:</strong> Show TypeScript examples only on the Developers page
						<ul className="mt-2 space-y-1 text-muted-foreground">
							<li>• TypeScript is the dominant language for modern web development</li>
							<li>• Shows the SDK has first-class TypeScript support</li>
							<li>• Reduces cognitive load (one language, clear examples)</li>
							<li>• Python examples can be on a separate SDK docs page</li>
						</ul>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong>Alternative:</strong> If targeting Python developers, create a separate section or page with
						Python examples
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Strategic decision to show TypeScript only. Reduces cognitive load and demonstrates first-class TypeScript support.',
			},
		},
	},
}

export const PythonExamplesAlternative: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersCodeExamples />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Alternative: Python Examples</h3>
				<div className="max-w-2xl mx-auto">
					<p className="text-sm text-muted-foreground mb-4">
						If targeting Python developers (data scientists, ML engineers), consider showing Python examples
						instead or in addition to TypeScript:
					</p>
					<div className="bg-background p-4 rounded-lg font-mono text-xs">
						<div className="text-muted-foreground"># Python example</div>
						<div className="mt-2">from llama_orch import invoke</div>
						<div className="mt-2">response = invoke(</div>
						<div className="pl-4">prompt="Generate a Python function",</div>
						<div className="pl-4">model="llama-3.1-70b"</div>
						<div>)</div>
					</div>
					<p className="text-xs text-muted-foreground mt-3">
						<strong>Recommendation:</strong> Create a separate page or section for Python examples rather than
						mixing languages on the same page.
					</p>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story:
					'Alternative approach: Show Python examples for data scientists and ML engineers. Recommend separate page to avoid mixing languages.',
			},
		},
	},
}

export const CodeAsMarketing: Story = {
	render: () => (
		<div className="space-y-8">
			<DevelopersCodeExamples />
			<div className="bg-muted p-8">
				<h3 className="text-xl font-bold mb-4 text-center">Code Examples as Marketing Tools</h3>
				<div className="max-w-3xl mx-auto space-y-4 text-sm">
					<p className="text-muted-foreground">
						<strong>CRITICAL:</strong> Code examples are not just documentation—they are marketing materials.
						Each example should demonstrate:
					</p>
					<div className="grid md:grid-cols-3 gap-4">
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Simplicity</strong>
							<p className="text-muted-foreground">
								Example 1 shows you can get started in 5 lines. No complex setup, no boilerplate.
							</p>
						</div>
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Power</strong>
							<p className="text-muted-foreground">
								Example 2 shows real workflow automation. Example 3 shows complex multi-step agents.
							</p>
						</div>
						<div className="bg-background p-4 rounded-lg">
							<strong className="block mb-2">Quality</strong>
							<p className="text-muted-foreground">
								Clean API design, composable utilities, TypeScript support. This is production-ready.
							</p>
						</div>
					</div>
					<div className="bg-background p-4 rounded-lg">
						<strong className="block mb-2">Marketing Strategy in Code Comments:</strong>
						<ul className="space-y-1 text-muted-foreground">
							<li>• Minimal comments (code should be self-documenting)</li>
							<li>• No tutorial comments (this is marketing, not a tutorial)</li>
							<li>• Variable names are descriptive (schema, code, review, suggestions)</li>
							<li>• Function names are clear (invoke, FileReader.read, Thread.create)</li>
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
					'Analysis of code examples as marketing tools. Each example demonstrates simplicity, power, and quality to convert developers.',
			},
		},
	},
}
