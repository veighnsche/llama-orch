import type { Meta, StoryObj } from '@storybook/react'
import { useToast } from './UseToast'
import { Button } from '@rbee/ui/atoms/Button'
import { Toaster } from '@rbee/ui/atoms/Toaster'

function UseToastDemo() {
	const { toast } = useToast()

	return (
		<div>
			<Button
				onClick={() => {
					toast({
						title: 'Notification',
						description: 'This is a toast notification using the useToast hook.',
					})
				}}
			>
				Show Toast
			</Button>
			<Toaster />
		</div>
	)
}

const meta: Meta = {
	title: 'Atoms/UseToast',
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj

export const Default: Story = {
	render: () => <UseToastDemo />,
}

function AllVariantsDemo() {
	const { toast } = useToast()

	return (
		<div className="flex flex-col gap-2">
			<Button
				onClick={() => {
					toast({
						title: 'Default Toast',
						description: 'This is a default toast notification.',
					})
				}}
			>
				Default
			</Button>
			<Button
				variant="destructive"
				onClick={() => {
					toast({
						variant: 'destructive',
						title: 'Error',
						description: 'There was a problem with your request.',
					})
				}}
			>
				Destructive
			</Button>
			<Toaster />
		</div>
	)
}

export const AllVariants: Story = {
	render: () => <AllVariantsDemo />,
}

function WithActionsDemo() {
	const { toast } = useToast()

	return (
		<div>
			<Button
				onClick={() => {
					toast({
						title: 'Event Scheduled',
						description: 'Your meeting has been scheduled for tomorrow.',
						action: (
							<Button size="sm" onClick={() => alert('Undo clicked')}>
								Undo
							</Button>
						),
					})
				}}
			>
				Show Toast with Action
			</Button>
			<Toaster />
		</div>
	)
}

export const WithActions: Story = {
	render: () => <WithActionsDemo />,
}

function WithDurationDemo() {
	const { toast } = useToast()

	return (
		<div className="flex flex-col gap-2">
			<Button
				onClick={() => {
					toast({
						title: 'Quick Toast',
						description: 'This toast will disappear quickly.',
					})
				}}
			>
				Default Duration
			</Button>
			<Toaster />
		</div>
	)
}

export const WithDuration: Story = {
	render: () => <WithDurationDemo />,
}
