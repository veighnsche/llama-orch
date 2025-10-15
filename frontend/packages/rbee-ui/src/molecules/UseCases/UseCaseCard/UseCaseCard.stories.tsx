import type { Meta, StoryObj } from '@storybook/react'
import { MessageSquare, FileText, Code, Search } from 'lucide-react'
import { UseCaseCard } from './UseCaseCard'

const meta: Meta<typeof UseCaseCard> = {
	title: 'Molecules/UseCases/UseCaseCard',
	component: UseCaseCard,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
## Overview
The UseCaseCard molecule displays a use case with scenario, solution, and highlights. Features icon, badge, and scroll-to-anchor support.

## Composition
This molecule is composed of:
- **IconBox**: Use case icon with color
- **Badge**: Optional badge (e.g., "Popular", "New")
- **Title**: Use case name
- **Scenario**: Problem description
- **Solution**: How rbee solves it
- **Highlights**: Key benefits with checkmarks

## When to Use
- Use case pages
- Feature demonstrations
- Solution showcases
- Customer success stories

## Used In
- **UseCasesPrimary**: Displays primary use cases (chatbots, document analysis, code generation, semantic search)
        `,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		icon: {
			control: false,
			description: 'Lucide icon component',
			table: {
				type: { summary: 'LucideIcon' },
				category: 'Content',
			},
		},
		color: {
			control: 'select',
			options: ['primary', 'chart-2', 'chart-3', 'chart-4'],
			description: 'Icon color',
			table: {
				type: { summary: "'primary' | 'chart-2' | 'chart-3' | 'chart-4'" },
				category: 'Appearance',
			},
		},
		title: {
			control: 'text',
			description: 'Use case title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		scenario: {
			control: 'text',
			description: 'Problem/scenario description',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		solution: {
			control: 'text',
			description: 'How rbee solves it',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		highlights: {
			control: 'object',
			description: 'Key benefits list',
			table: {
				type: { summary: 'string[]' },
				category: 'Content',
			},
		},
		anchor: {
			control: 'text',
			description: 'Optional anchor ID',
			table: {
				type: { summary: 'string' },
				category: 'Behavior',
			},
		},
		badge: {
			control: 'text',
			description: 'Optional badge text',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof UseCaseCard>

export const Default: Story = {
	args: {
		icon: MessageSquare,
		color: 'primary',
		title: 'Customer Support Chatbots',
		scenario: 'Your support team is overwhelmed with repetitive questions, and cloud AI services expose customer data.',
		solution: 'Deploy a private chatbot that handles common queries 24/7 while keeping all conversations on your infrastructure.',
		highlights: [
			'Instant responses to common questions',
			'Zero customer data leaves your servers',
			'Scales to handle thousands of conversations',
		],
	},
}

export const WithIcon: Story = {
	args: {
		icon: FileText,
		color: 'chart-2',
		title: 'Document Analysis',
		scenario: 'Legal teams spend hours reviewing contracts and compliance documents manually.',
		solution: 'Automate document review with AI that understands legal language and flags potential issues.',
		highlights: [
			'Review documents 10x faster',
			'Identify risks and compliance issues',
			'Maintain attorney-client privilege',
		],
		badge: 'Popular',
	},
}

export const WithOutcome: Story = {
	args: {
		icon: Code,
		color: 'chart-3',
		title: 'Code Generation & Review',
		scenario: 'Developers need AI assistance but can\'t send proprietary code to external services.',
		solution: 'Run code-aware LLMs on your infrastructure to assist with development without exposing IP.',
		highlights: [
			'Generate boilerplate and tests',
			'Review code for bugs and security',
			'Your code never leaves your network',
			'Supports multiple languages',
		],
		anchor: 'code-generation',
	},
}

export const InUseCasesContext: Story = {
	render: () => (
		<div className="w-full max-w-6xl">
			<div className="mb-4 text-sm text-muted-foreground">
				Example: UseCaseCard in UseCasesPrimary organism
			</div>
			<div className="grid gap-6 md:grid-cols-2">
				<UseCaseCard
					icon={MessageSquare}
					color="primary"
					title="Customer Support Chatbots"
					scenario="Your support team is overwhelmed with repetitive questions, and cloud AI services expose customer data."
					solution="Deploy a private chatbot that handles common queries 24/7 while keeping all conversations on your infrastructure."
					highlights={[
						'Instant responses to common questions',
						'Zero customer data leaves your servers',
						'Scales to handle thousands of conversations',
					]}
					badge="Most Popular"
					anchor="chatbots"
				/>
				<UseCaseCard
					icon={Search}
					color="chart-4"
					title="Semantic Search"
					scenario="Traditional keyword search misses relevant documents and frustrates users."
					solution="Implement AI-powered semantic search that understands meaning, not just keywords."
					highlights={[
						'Find documents by meaning, not exact words',
						'Works across multiple languages',
						'Private embeddings stay on your servers',
					]}
					anchor="search"
				/>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'UseCaseCard as used in the UseCasesPrimary organism, showing two primary use cases.',
			},
		},
	},
}
