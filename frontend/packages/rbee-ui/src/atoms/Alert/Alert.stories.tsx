// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Terminal, AlertCircle, CheckCircle2, Info } from 'lucide-react'
import { Alert, AlertTitle, AlertDescription } from './Alert'

const meta: Meta<typeof Alert> = {
	title: 'Atoms/Alert',
	component: Alert,
	parameters: {
		layout: 'padded',
	},
	tags: ['autodocs'],
	argTypes: {
		variant: {
			control: 'select',
			options: ['default', 'destructive', 'success', 'primary', 'info', 'warning'],
			description: 'Visual style variant of the alert',
		},
	},
}

export default meta
type Story = StoryObj<typeof Alert>

/**
 * ## Overview
 * Alert is a non-intrusive notification component used to display important information
 * to users without blocking their workflow. Built with Radix UI primitives.
 *
 * ## When to Use
 * - Display system status messages
 * - Show validation errors or warnings
 * - Provide contextual information
 * - Highlight important notices
 *
 * ## Used In
 * - Form validation feedback
 * - System notifications
 * - Error boundaries
 * - Status indicators
 */

export const Default: Story = {
	render: () => (
		<Alert>
			<Terminal />
			<AlertTitle>Heads up!</AlertTitle>
			<AlertDescription>You can add components to your app using the CLI.</AlertDescription>
		</Alert>
	),
}

export const AllVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-4 max-w-xl">
			<Alert variant="default">
				<Info />
				<AlertTitle>Information</AlertTitle>
				<AlertDescription>This is a default informational alert message.</AlertDescription>
			</Alert>
			<Alert variant="destructive">
				<AlertCircle />
				<AlertTitle>Error</AlertTitle>
				<AlertDescription>This is a destructive alert for errors or critical warnings.</AlertDescription>
			</Alert>
			<Alert variant="success">
				<CheckCircle2 />
				<AlertTitle>Success</AlertTitle>
				<AlertDescription>Your data never leaves your infrastructure.</AlertDescription>
			</Alert>
			<Alert variant="primary">
				<Info />
				<AlertTitle>Primary</AlertTitle>
				<AlertDescription>Lightning-fast inference with GPU acceleration.</AlertDescription>
			</Alert>
			<Alert variant="info">
				<Info />
				<AlertTitle>Info</AlertTitle>
				<AlertDescription>GDPR compliant by design.</AlertDescription>
			</Alert>
			<Alert variant="warning">
				<AlertCircle />
				<AlertTitle>Warning</AlertTitle>
				<AlertDescription>Bank-grade security with zero-trust architecture.</AlertDescription>
			</Alert>
		</div>
	),
}

export const WithIcon: Story = {
	render: () => (
		<div className="flex flex-col gap-4 max-w-xl">
			<Alert>
				<CheckCircle2 />
				<AlertTitle>Success</AlertTitle>
				<AlertDescription>Your changes have been saved successfully.</AlertDescription>
			</Alert>
			<Alert>
				<Terminal />
				<AlertTitle>Command Line</AlertTitle>
				<AlertDescription>Run `npm install @rbee/ui` to install the package.</AlertDescription>
			</Alert>
		</div>
	),
}

export const WithTitle: Story = {
	render: () => (
		<div className="flex flex-col gap-4 max-w-xl">
			<Alert>
				<AlertTitle>Simple Title</AlertTitle>
				<AlertDescription>This alert has a title and description.</AlertDescription>
			</Alert>
			<Alert variant="destructive">
				<AlertCircle />
				<AlertTitle>Critical Error</AlertTitle>
				<AlertDescription>
					Your session has expired. Please log in again to continue. This is a longer description to show how
					text wraps in the alert component.
				</AlertDescription>
			</Alert>
		</div>
	),
}
