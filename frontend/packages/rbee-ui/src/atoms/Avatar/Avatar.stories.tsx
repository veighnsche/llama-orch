// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Avatar, AvatarFallback, AvatarImage } from './Avatar'

const meta: Meta<typeof Avatar> = {
	title: 'Atoms/Avatar',
	component: Avatar,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Avatar>

/**
 * ## Overview
 * Avatar displays a user's profile picture or initials in a circular container.
 * Built with Radix UI Avatar primitives with automatic fallback support.
 *
 * ## When to Use
 * - Display user profile pictures
 * - Show team member avatars
 * - Represent users in comments or messages
 * - Create avatar groups
 *
 * ## Used In
 * - User profiles
 * - Comment sections
 * - Team member lists
 * - Navigation headers
 */

export const Default: Story = {
	render: () => (
		<Avatar>
			<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
			<AvatarFallback>CN</AvatarFallback>
		</Avatar>
	),
}

export const WithFallback: Story = {
	render: () => (
		<div className="flex gap-4">
			<Avatar>
				<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
				<AvatarFallback>CN</AvatarFallback>
			</Avatar>
			<Avatar>
				<AvatarImage src="/broken-image.jpg" alt="@broken" />
				<AvatarFallback>BK</AvatarFallback>
			</Avatar>
			<Avatar>
				<AvatarFallback>JD</AvatarFallback>
			</Avatar>
		</div>
	),
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex items-center gap-4">
			<Avatar className="size-6">
				<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
				<AvatarFallback className="text-xs">XS</AvatarFallback>
			</Avatar>
			<Avatar className="size-8">
				<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
				<AvatarFallback className="text-xs">SM</AvatarFallback>
			</Avatar>
			<Avatar className="size-10">
				<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
				<AvatarFallback>MD</AvatarFallback>
			</Avatar>
			<Avatar className="size-12">
				<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
				<AvatarFallback>LG</AvatarFallback>
			</Avatar>
			<Avatar className="size-16">
				<AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
				<AvatarFallback className="text-lg">XL</AvatarFallback>
			</Avatar>
		</div>
	),
}

export const Group: Story = {
	render: () => (
		<div className="flex flex-col gap-6">
			<div>
				<p className="text-sm font-medium mb-3">Team Members</p>
				<div className="flex -space-x-2">
					<Avatar className="border-2 border-background">
						<AvatarImage src="https://github.com/shadcn.png" alt="User 1" />
						<AvatarFallback>U1</AvatarFallback>
					</Avatar>
					<Avatar className="border-2 border-background">
						<AvatarImage src="https://github.com/vercel.png" alt="User 2" />
						<AvatarFallback>U2</AvatarFallback>
					</Avatar>
					<Avatar className="border-2 border-background">
						<AvatarFallback>U3</AvatarFallback>
					</Avatar>
					<Avatar className="border-2 border-background">
						<AvatarFallback>U4</AvatarFallback>
					</Avatar>
					<Avatar className="border-2 border-background">
						<AvatarFallback className="text-xs">+5</AvatarFallback>
					</Avatar>
				</div>
			</div>
			<div>
				<p className="text-sm font-medium mb-3">With Names</p>
				<div className="flex gap-4">
					<div className="flex flex-col items-center gap-2">
						<Avatar>
							<AvatarImage src="https://github.com/shadcn.png" alt="Alice" />
							<AvatarFallback>AL</AvatarFallback>
						</Avatar>
						<span className="text-xs">Alice</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<Avatar>
							<AvatarFallback>BM</AvatarFallback>
						</Avatar>
						<span className="text-xs">Bob</span>
					</div>
					<div className="flex flex-col items-center gap-2">
						<Avatar>
							<AvatarFallback>CD</AvatarFallback>
						</Avatar>
						<span className="text-xs">Charlie</span>
					</div>
				</div>
			</div>
		</div>
	),
}
