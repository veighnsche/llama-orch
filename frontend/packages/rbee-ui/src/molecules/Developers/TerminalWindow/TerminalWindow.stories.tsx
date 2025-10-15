import type { Meta, StoryObj } from '@storybook/react'
import { TerminalWindow } from './TerminalWindow'

const meta: Meta<typeof TerminalWindow> = {
	title: 'Molecules/Developers/TerminalWindow',
	component: TerminalWindow,
	parameters: {
		layout: 'padded',
		docs: {
			description: {
				component: `
## Overview
TerminalWindow is a styled terminal/code window molecule that mimics macOS terminal aesthetics. It's used to display code snippets, command-line examples, and terminal output.

## Composition
This molecule is composed of:
- **Window chrome**: macOS-style traffic lights (red, yellow, green)
- **Title bar**: Optional title with muted background
- **Content area**: Monospace font for code/terminal content
- **Shadow**: Elevated shadow for depth

## When to Use
- Code examples (CLI commands)
- Terminal output (logs, results)
- Installation instructions (setup steps)
- API responses (JSON output)
- Developer documentation (technical examples)

## Variants
- **terminal**: Terminal-style output (default)
- **code**: Code snippet display
- **output**: Command output display

## Used In Commercial Site
Used in:
- HeroSection (quick start command)
- DevelopersCodeExamples (API examples)
- HowItWorksSection (setup instructions)
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		title: {
			control: 'text',
			description: 'Terminal title',
			table: {
				type: { summary: 'string' },
				category: 'Content',
			},
		},
		variant: {
			control: 'select',
			options: ['terminal', 'code', 'output'],
			description: 'Terminal variant',
			table: {
				type: { summary: "'terminal' | 'code' | 'output'" },
				defaultValue: { summary: 'terminal' },
				category: 'Appearance',
			},
		},
		children: {
			control: false,
			description: 'Terminal content',
			table: {
				type: { summary: 'ReactNode' },
				category: 'Content',
			},
		},
	},
}

export default meta
type Story = StoryObj<typeof TerminalWindow>

export const Default: Story = {
	args: {
		title: 'bash',
		children: (
			<div className="text-foreground">
				<div className="text-muted-foreground">$ npm install @rbee/sdk</div>
				<div className="mt-2">✓ Installed @rbee/sdk@1.0.0</div>
			</div>
		),
	},
}

export const WithCommand: Story = {
	args: {
		title: 'terminal',
		children: (
			<div className="space-y-2">
				<div className="text-chart-3">$ rbee init</div>
				<div className="text-muted-foreground">✓ Created .rbee.toml</div>
				<div className="text-muted-foreground">✓ Detected GPU: NVIDIA RTX 4090</div>
				<div className="text-muted-foreground">✓ Worker registered successfully</div>
				<div className="mt-2 text-chart-3">$ rbee start</div>
				<div className="text-muted-foreground">🐝 Worker started on port 8080</div>
			</div>
		),
	},
}

export const Streaming: Story = {
	render: () => (
		<div className="space-y-6 w-full max-w-2xl">
			<div>
				<h3 className="text-lg font-semibold mb-3">Installation</h3>
				<TerminalWindow title="bash">
					<div className="space-y-1">
						<div className="text-chart-3">$ curl -fsSL https://rbee.nl/install.sh | bash</div>
						<div className="text-muted-foreground">Downloading rbee...</div>
						<div className="text-muted-foreground">Installing to /usr/local/bin...</div>
						<div className="text-chart-3">✓ rbee installed successfully</div>
					</div>
				</TerminalWindow>
			</div>
			<div>
				<h3 className="text-lg font-semibold mb-3">Quick Start</h3>
				<TerminalWindow title="terminal">
					<div className="space-y-1">
						<div className="text-chart-3">$ rbee init</div>
						<div className="text-muted-foreground">✓ Configuration created</div>
						<div className="text-chart-3">$ rbee start</div>
						<div className="text-muted-foreground">🐝 Worker ready at http://localhost:8080</div>
					</div>
				</TerminalWindow>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'Multiple terminal windows showing installation and setup flow.',
			},
		},
	},
}

export const InHeroContext: Story = {
	render: () => (
		<div className="w-full max-w-4xl">
			<div className="mb-8 text-center">
				<h2 className="text-3xl font-bold mb-2">Get Started in Minutes</h2>
				<p className="text-muted-foreground">Install rbee and start hosting LLMs on your infrastructure</p>
			</div>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
				<div>
					<h3 className="text-sm font-semibold mb-3">1. Install</h3>
					<TerminalWindow title="bash">
						<div className="text-chart-3">$ npm install -g @rbee/cli</div>
					</TerminalWindow>
				</div>
				<div>
					<h3 className="text-sm font-semibold mb-3">2. Initialize</h3>
					<TerminalWindow title="bash">
						<div className="text-chart-3">$ rbee init</div>
					</TerminalWindow>
				</div>
				<div>
					<h3 className="text-sm font-semibold mb-3">3. Start Worker</h3>
					<TerminalWindow title="bash">
						<div className="text-chart-3">$ rbee start</div>
					</TerminalWindow>
				</div>
				<div>
					<h3 className="text-sm font-semibold mb-3">4. Deploy Model</h3>
					<TerminalWindow title="bash">
						<div className="text-chart-3">$ rbee deploy llama-3-8b</div>
					</TerminalWindow>
				</div>
			</div>
		</div>
	),
	parameters: {
		docs: {
			description: {
				story: 'TerminalWindow as used in HeroSection, showing quick start steps.',
			},
		},
	},
}
