// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Button } from '@rbee/ui/atoms/Button'
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
	AlertDialogTrigger,
} from './AlertDialog'

const meta: Meta<typeof AlertDialog> = {
	title: 'Atoms/AlertDialog',
	component: AlertDialog,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof AlertDialog>

/**
 * ## Overview
 * AlertDialog is a modal dialog that interrupts the user with important content
 * and expects a response. Built with Radix UI AlertDialog primitives.
 *
 * ## When to Use
 * - Confirm destructive actions (delete, remove)
 * - Display critical warnings
 * - Request user confirmation
 * - Show important information that requires acknowledgment
 *
 * ## Used In
 * - Delete confirmations
 * - Logout flows
 * - Data loss warnings
 * - Critical system messages
 */

export const Default: Story = {
	render: () => (
		<AlertDialog>
			<AlertDialogTrigger asChild>
				<Button variant="outline">Show Dialog</Button>
			</AlertDialogTrigger>
			<AlertDialogContent>
				<AlertDialogHeader>
					<AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
					<AlertDialogDescription>
						This action cannot be undone. This will permanently delete your account and remove your data from our
						servers.
					</AlertDialogDescription>
				</AlertDialogHeader>
				<AlertDialogFooter>
					<AlertDialogCancel>Cancel</AlertDialogCancel>
					<AlertDialogAction>Continue</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	),
}

export const Destructive: Story = {
	render: () => (
		<AlertDialog>
			<AlertDialogTrigger asChild>
				<Button variant="destructive">Delete Account</Button>
			</AlertDialogTrigger>
			<AlertDialogContent>
				<AlertDialogHeader>
					<AlertDialogTitle>Delete Account</AlertDialogTitle>
					<AlertDialogDescription>
						This will permanently delete your account. All your data, models, and configurations will be lost. This
						action cannot be undone.
					</AlertDialogDescription>
				</AlertDialogHeader>
				<AlertDialogFooter>
					<AlertDialogCancel>Cancel</AlertDialogCancel>
					<AlertDialogAction className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
						Delete Account
					</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	),
}

export const WithForm: Story = {
	render: () => (
		<AlertDialog>
			<AlertDialogTrigger asChild>
				<Button>Confirm Action</Button>
			</AlertDialogTrigger>
			<AlertDialogContent>
				<AlertDialogHeader>
					<AlertDialogTitle>Confirm your action</AlertDialogTitle>
					<AlertDialogDescription>
						Please type "DELETE" to confirm this destructive action.
					</AlertDialogDescription>
				</AlertDialogHeader>
				<input
					type="text"
					placeholder="Type DELETE"
					className="border rounded px-3 py-2 text-sm"
				/>
				<AlertDialogFooter>
					<AlertDialogCancel>Cancel</AlertDialogCancel>
					<AlertDialogAction>Confirm</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	),
}

export const LongContent: Story = {
	render: () => (
		<AlertDialog>
			<AlertDialogTrigger asChild>
				<Button variant="outline">View Terms</Button>
			</AlertDialogTrigger>
			<AlertDialogContent className="max-h-[80vh] overflow-y-auto">
				<AlertDialogHeader>
					<AlertDialogTitle>Terms and Conditions</AlertDialogTitle>
					<AlertDialogDescription>
						Please read and accept our terms and conditions before continuing.
					</AlertDialogDescription>
				</AlertDialogHeader>
				<div className="text-sm space-y-4">
					<p>
						Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore
						et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.
					</p>
					<p>
						Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
						pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit
						anim id est laborum.
					</p>
					<p>
						Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium,
						totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae
						dicta sunt explicabo.
					</p>
				</div>
				<AlertDialogFooter>
					<AlertDialogCancel>Decline</AlertDialogCancel>
					<AlertDialogAction>Accept</AlertDialogAction>
				</AlertDialogFooter>
			</AlertDialogContent>
		</AlertDialog>
	),
}
