// Created by: TEAM-007
import type { Meta, StoryObj } from '@storybook/react'
import { Input } from './Input'

const meta: Meta<typeof Input> = {
	title: 'Atoms/Input',
	component: Input,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		type: {
			control: 'select',
			options: ['text', 'email', 'password', 'number', 'search', 'tel', 'url'],
			description: 'HTML input type',
		},
		disabled: {
			control: 'boolean',
			description: 'Disable the input',
		},
	},
}

export default meta
type Story = StoryObj<typeof Input>

/**
 * ## Overview
 * Input is a text input field with consistent styling and built-in validation states.
 * Supports all standard HTML input types with focus and error states.
 *
 * ## When to Use
 * - Collect user text input
 * - Forms and settings
 * - Search functionality
 * - Email capture
 * - Authentication flows
 *
 * ## Used In
 * - EmailCaptureForm
 * - ContactForm
 * - SearchBar
 */

export const Default: Story = {
	args: {
		placeholder: 'Enter text...',
	},
}

export const WithIcon: Story = {
	render: () => (
		<div className="w-[300px] space-y-4">
			{/* Search with icon */}
			<div className="relative">
				<svg
					className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground"
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path
						strokeLinecap="round"
						strokeLinejoin="round"
						strokeWidth={2}
						d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
					/>
				</svg>
				<Input placeholder="Search models..." className="pl-9" />
			</div>

			{/* Email with icon */}
			<div className="relative">
				<svg
					className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground"
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path
						strokeLinecap="round"
						strokeLinejoin="round"
						strokeWidth={2}
						d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
					/>
				</svg>
				<Input type="email" placeholder="your@email.com" className="pl-9" />
			</div>

			{/* Password with icon */}
			<div className="relative">
				<svg
					className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground"
					fill="none"
					stroke="currentColor"
					viewBox="0 0 24 24"
				>
					<path
						strokeLinecap="round"
						strokeLinejoin="round"
						strokeWidth={2}
						d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
					/>
				</svg>
				<Input type="password" placeholder="Password" className="pl-9" />
			</div>
		</div>
	),
}

export const AllStates: Story = {
	render: () => (
		<div className="w-[300px] space-y-4">
			<div>
				<label className="text-sm font-medium mb-1.5 block">Default</label>
				<Input placeholder="Enter text..." />
			</div>

			<div>
				<label className="text-sm font-medium mb-1.5 block">Focused</label>
				<Input placeholder="Click to focus" autoFocus />
			</div>

			<div>
				<label className="text-sm font-medium mb-1.5 block">Disabled</label>
				<Input placeholder="Disabled input" disabled />
			</div>

			<div>
				<label className="text-sm font-medium mb-1.5 block">Error</label>
				<Input placeholder="Invalid input" aria-invalid="true" />
				<p className="text-xs text-destructive mt-1">This field is required</p>
			</div>

			<div>
				<label className="text-sm font-medium mb-1.5 block">With Value</label>
				<Input defaultValue="Existing value" />
			</div>

			<div>
				<label className="text-sm font-medium mb-1.5 block">Different Types</label>
				<div className="space-y-2">
					<Input type="email" placeholder="email@example.com" />
					<Input type="password" placeholder="Password" />
					<Input type="number" placeholder="123" />
					<Input type="search" placeholder="Search..." />
				</div>
			</div>
		</div>
	),
}

export const InEmailCapture: Story = {
	render: () => (
		<div className="max-w-md mx-auto p-6 border rounded-lg">
			<div className="text-center mb-6">
				<h3 className="text-2xl font-bold mb-2">Stay Updated</h3>
				<p className="text-muted-foreground">Get the latest updates on new models and features</p>
			</div>

			<form className="space-y-4">
				<div>
					<label htmlFor="name" className="text-sm font-medium mb-1.5 block">
						Name
					</label>
					<Input id="name" placeholder="John Doe" />
				</div>

				<div>
					<label htmlFor="email" className="text-sm font-medium mb-1.5 block">
						Email
					</label>
					<Input id="email" type="email" placeholder="john@example.com" />
				</div>

				<div>
					<label htmlFor="company" className="text-sm font-medium mb-1.5 block">
						Company (optional)
					</label>
					<Input id="company" placeholder="Acme Inc." />
				</div>

				<button
					type="submit"
					className="w-full bg-primary text-primary-foreground rounded-md px-4 py-2 font-medium hover:bg-primary/90 transition-colors"
				>
					Subscribe
				</button>

				<p className="text-xs text-center text-muted-foreground">
					We respect your privacy. Unsubscribe at any time.
				</p>
			</form>
		</div>
	),
}
