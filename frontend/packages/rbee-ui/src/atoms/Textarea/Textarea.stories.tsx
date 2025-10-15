// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { useState } from 'react'
import { Textarea } from './Textarea'

const meta: Meta<typeof Textarea> = {
	title: 'Atoms/Textarea',
	component: Textarea,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
A textarea component for multi-line text input.

## Features
- Auto-sizing with field-sizing-content
- Minimum height of 64px (16 units)
- Focus visible states with ring
- Disabled state support
- Placeholder support
- Dark mode compatible

## Used In
- Forms
- Comment sections
- Message inputs
- Configuration editors
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		placeholder: {
			control: 'text',
			description: 'Placeholder text',
		},
		disabled: {
			control: 'boolean',
			description: 'Disable the textarea',
		},
		rows: {
			control: 'number',
			description: 'Number of visible text rows',
		},
	},
}

export default meta
type Story = StoryObj<typeof Textarea>

export const Default: Story = {
	args: {
		placeholder: 'Type your message here...',
	},
}

export const WithCounter: Story = {
	render: () => {
		const [value, setValue] = useState('')
		const maxLength = 500

		return (
			<div className="w-full max-w-md space-y-2">
				<label className="text-sm font-medium">Description</label>
				<Textarea
					placeholder="Describe your project..."
					value={value}
					onChange={(e) => setValue(e.target.value)}
					maxLength={maxLength}
				/>
				<div className="flex justify-between text-xs text-muted-foreground">
					<span>Maximum {maxLength} characters</span>
					<span>
						{value.length}/{maxLength}
					</span>
				</div>
			</div>
		)
	},
}

export const Resizable: Story = {
	render: () => (
		<div className="flex flex-col gap-6 w-full max-w-md">
			<div className="space-y-2">
				<label className="text-sm font-medium">Auto-resize (default)</label>
				<Textarea placeholder="This textarea auto-resizes as you type..." />
				<p className="text-xs text-muted-foreground">Uses field-sizing-content for automatic height</p>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">Fixed height</label>
				<Textarea
					placeholder="This textarea has a fixed height..."
					className="resize-none h-32"
				/>
				<p className="text-xs text-muted-foreground">Uses resize-none and fixed height</p>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">Vertical resize only</label>
				<Textarea
					placeholder="You can resize this vertically..."
					className="resize-y min-h-24"
				/>
				<p className="text-xs text-muted-foreground">Users can adjust height manually</p>
			</div>
		</div>
	),
}

export const DisabledState: Story = {
	render: () => (
		<div className="flex flex-col gap-4 w-full max-w-md">
			<div className="space-y-2">
				<label className="text-sm font-medium">Enabled</label>
				<Textarea placeholder="You can type here..." />
			</div>
			<div className="space-y-2">
				<label className="text-sm font-medium text-muted-foreground">Disabled</label>
				<Textarea
					placeholder="This is disabled..."
					disabled
					value="This content cannot be edited"
				/>
			</div>
		</div>
	),
}

export const InForm: Story = {
	render: () => (
		<form className="w-full max-w-2xl space-y-6 p-6 border border-border rounded-lg">
			<div>
				<h3 className="text-lg font-semibold mb-2">Submit Feedback</h3>
				<p className="text-sm text-muted-foreground">Help us improve by sharing your thoughts</p>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">
					Subject <span className="text-destructive">*</span>
				</label>
				<input
					type="text"
					required
					className="w-full px-3 py-2 border border-input rounded-md bg-transparent text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
					placeholder="Brief summary of your feedback"
				/>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">
					Feedback <span className="text-destructive">*</span>
				</label>
				<Textarea
					required
					placeholder="Tell us what you think..."
					className="min-h-32"
				/>
				<p className="text-xs text-muted-foreground">
					Please be as detailed as possible. This helps us understand your needs better.
				</p>
			</div>

			<div className="space-y-2">
				<label className="text-sm font-medium">Additional Notes (optional)</label>
				<Textarea
					placeholder="Any other information you'd like to share..."
					className="min-h-24"
				/>
			</div>

			<div className="flex gap-3">
				<button
					type="submit"
					className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors"
				>
					Submit Feedback
				</button>
				<button
					type="button"
					className="px-4 py-2 border border-input rounded-md text-sm font-medium hover:bg-muted/40 transition-colors"
				>
					Cancel
				</button>
			</div>
		</form>
	),
}

export const CodeEditor: Story = {
	render: () => (
		<div className="w-full max-w-3xl space-y-4 p-6 border border-border rounded-lg">
			<div className="flex items-center justify-between">
				<h3 className="text-sm font-semibold">System Prompt</h3>
				<div className="flex gap-2">
					<button className="px-3 py-1 text-xs border border-input rounded-md hover:bg-muted/40 transition-colors">
						Reset
					</button>
					<button className="px-3 py-1 text-xs bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
						Save
					</button>
				</div>
			</div>
			<Textarea
				className="font-mono text-sm min-h-48 resize-y"
				defaultValue={`You are a helpful AI assistant specialized in software development.

Your responses should be:
- Clear and concise
- Technically accurate
- Include code examples when relevant
- Follow best practices

Always explain your reasoning and provide context for your suggestions.`}
			/>
			<div className="flex items-center justify-between text-xs text-muted-foreground">
				<span>Markdown supported</span>
				<span>Last saved: 2 minutes ago</span>
			</div>
		</div>
	),
}

export const ChatInput: Story = {
	render: () => (
		<div className="w-full max-w-3xl">
			<div className="border border-border rounded-lg overflow-hidden">
				<div className="p-4 border-b border-border bg-muted/20">
					<h3 className="text-sm font-semibold">Chat with Llama 3.1 8B</h3>
				</div>
				<div className="p-4 min-h-[300px] max-h-[400px] overflow-y-auto space-y-4">
					<div className="flex gap-3">
						<div className="size-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-medium">
							AI
						</div>
						<div className="flex-1 space-y-1">
							<p className="text-sm">Hello! How can I help you today?</p>
						</div>
					</div>
				</div>
				<div className="p-4 border-t border-border">
					<div className="flex gap-2">
						<Textarea
							placeholder="Type your message... (Shift+Enter for new line)"
							className="min-h-12 max-h-32 resize-none"
							onKeyDown={(e) => {
								if (e.key === 'Enter' && !e.shiftKey) {
									e.preventDefault()
									// Handle send
								}
							}}
						/>
						<button className="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 transition-colors self-end">
							Send
						</button>
					</div>
					<p className="text-xs text-muted-foreground mt-2">
						Press Enter to send, Shift+Enter for new line
					</p>
				</div>
			</div>
		</div>
	),
}
