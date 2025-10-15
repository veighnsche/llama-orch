import type { Meta, StoryObj } from '@storybook/react'
import { Toast, ToastTitle, ToastDescription, ToastAction, ToastProvider, ToastViewport } from './Toast'
import { Button } from '@rbee/ui/atoms/Button'
import { useState } from 'react'

const meta: Meta<typeof Toast> = {
	title: 'Atoms/Toast',
	component: Toast,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		variant: {
			control: 'select',
			options: ['default', 'destructive'],
			description: 'Toast variant',
		},
	},
}

export default meta
type Story = StoryObj<typeof Toast>

export const Default: Story = {
	render: () => {
		const [open, setOpen] = useState(false)
		return (
			<ToastProvider>
				<Button onClick={() => setOpen(true)}>Show Toast</Button>
				<Toast open={open} onOpenChange={setOpen}>
					<ToastTitle>Notification</ToastTitle>
					<ToastDescription>This is a default toast notification.</ToastDescription>
				</Toast>
				<ToastViewport />
			</ToastProvider>
		)
	},
}

export const AllVariants: Story = {
	render: () => {
		const [open1, setOpen1] = useState(false)
		const [open2, setOpen2] = useState(false)
		return (
			<ToastProvider>
				<div className="flex flex-col gap-2">
					<Button onClick={() => setOpen1(true)}>Show Default Toast</Button>
					<Button variant="destructive" onClick={() => setOpen2(true)}>
						Show Destructive Toast
					</Button>
				</div>
				<Toast open={open1} onOpenChange={setOpen1}>
					<ToastTitle>Success</ToastTitle>
					<ToastDescription>Your changes have been saved successfully.</ToastDescription>
				</Toast>
				<Toast open={open2} onOpenChange={setOpen2} variant="destructive">
					<ToastTitle>Error</ToastTitle>
					<ToastDescription>There was a problem with your request.</ToastDescription>
				</Toast>
				<ToastViewport />
			</ToastProvider>
		)
	},
}

export const WithAction: Story = {
	render: () => {
		const [open, setOpen] = useState(false)
		return (
			<ToastProvider>
				<Button onClick={() => setOpen(true)}>Show Toast with Action</Button>
				<Toast open={open} onOpenChange={setOpen}>
					<ToastTitle>Event Scheduled</ToastTitle>
					<ToastDescription>Your meeting has been scheduled for tomorrow at 10 AM.</ToastDescription>
					<ToastAction altText="Undo action" onClick={() => alert('Undo clicked')}>
						Undo
					</ToastAction>
				</Toast>
				<ToastViewport />
			</ToastProvider>
		)
	},
}

export const AllPositions: Story = {
	render: () => {
		const [open, setOpen] = useState(false)
		return (
			<ToastProvider>
				<Button onClick={() => setOpen(true)}>Show Toast</Button>
				<Toast open={open} onOpenChange={setOpen}>
					<ToastTitle>Position Demo</ToastTitle>
					<ToastDescription>
						This toast appears in the default position (bottom-right on desktop, top on mobile).
					</ToastDescription>
				</Toast>
				<ToastViewport />
			</ToastProvider>
		)
	},
}
