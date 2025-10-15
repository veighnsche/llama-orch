import type { Meta, StoryObj } from '@storybook/react'
import { CodeBlock } from './CodeBlock'

const meta: Meta<typeof CodeBlock> = {
	title: 'Molecules/CodeBlock',
	component: CodeBlock,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The CodeBlock molecule displays code with syntax highlighting, copy button, optional title, line numbers, and line highlighting.

## Composition
This molecule is composed of:
- **Header**: Optional title and language badge
- **Copy Button**: Click to copy code
- **Code Area**: Monospace font with optional line numbers
- **Highlighting**: Optional line highlighting

## When to Use
- Code examples
- API documentation
- Configuration snippets
- Command-line examples
- Tutorial content

## Used In
- **FeaturesSection**: Displays code examples for API usage and integration
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		code: {
			control: 'text',
			description: 'Code content',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		language: {
			control: 'text',
			description: 'Programming language',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		title: {
			control: 'text',
			description: 'Optional title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		copyable: {
			control: 'boolean',
			description: 'Show copy button',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'true' },
				category: 'Behavior',
			},
		},
		showLineNumbers: {
			control: 'boolean',
			description: 'Show line numbers',
			table: {
				type: { summary: 'boolean' },
				defaultValue: { summary: 'false' },
				category: 'Appearance',
			},
		},
		highlight: {
			control: 'object',
			description: 'Line numbers to highlight',
			table: {
				type: { summary: 'number[]' },
				category: 'Appearance',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof CodeBlock>

const pythonCode = `import requests

response = requests.post(
    "https://api.rbee.nl/v1/inference",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "model": "llama-3.1-8b",
        "prompt": "Explain quantum computing",
        "max_tokens": 500
    }
)

print(response.json())`

const bashCode = `curl -X POST https://api.rbee.nl/v1/inference \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "llama-3.1-8b",
    "prompt": "Explain quantum computing",
    "max_tokens": 500
  }'`

const typescriptCode = `import { RbeeClient } from '@rbee/sdk'

const client = new RbeeClient({
  apiKey: process.env.RBEE_API_KEY,
  baseUrl: 'https://api.rbee.nl'
})

const response = await client.inference({
  model: 'llama-3.1-8b',
  prompt: 'Explain quantum computing',
  maxTokens: 500
})`

export const Default: Story = {
	args: {
		code: pythonCode,
		language: 'python',
		title: 'example.py',
	},
}

export const AllLanguages: Story = {
	render: () => (
		<div className="flex flex-col gap-6 p-8 max-w-3xl">
			<h3 className="text-lg font-semibold text-foreground">All Languages</h3>
			<CodeBlock
				code={pythonCode}
				language="python"
				title="example.py"
			/>
			<CodeBlock
				code={bashCode}
				language="bash"
				title="curl-example.sh"
			/>
			<CodeBlock
				code={typescriptCode}
				language="typescript"
				title="example.ts"
			/>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'CodeBlock with different programming languages.',
			},
		},
	},
}

export const WithCopy: Story = {
	args: {
		code: typescriptCode,
		language: 'typescript',
		title: 'Quick Start',
		copyable: true,
		showLineNumbers: true,
		highlight: [3, 4, 5],
	},
}

export const InFeaturesContext: Story = {
	render: () => (
		<div className="w-full max-w-5xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: CodeBlock in FeaturesSection organism
			</div>
			<div className="rounded-2xl border border-border bg-card p-8">
				<div className="mb-6">
					<h2 className="text-2xl font-bold text-foreground mb-2">Simple API Integration</h2>
					<p className="text-muted-foreground">
						Get started in minutes with our developer-friendly API. Works with any language.
					</p>
				</div>
				<div className="grid gap-6 lg:grid-cols-2">
					<div>
						<h3 className="text-sm font-semibold text-foreground mb-3">Python</h3>
						<CodeBlock
							code={pythonCode}
							language="python"
							title="example.py"
						/>
					</div>
					<div>
						<h3 className="text-sm font-semibold text-foreground mb-3">cURL</h3>
						<CodeBlock
							code={bashCode}
							language="bash"
							title="curl-example.sh"
						/>
					</div>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'CodeBlock as used in the FeaturesSection organism, showing API examples in multiple languages.',
			},
		},
	},
}
