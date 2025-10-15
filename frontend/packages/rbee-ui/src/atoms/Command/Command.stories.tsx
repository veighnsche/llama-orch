// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Search, Settings, User, Mail, Calendar } from 'lucide-react'

const meta: Meta = {
	title: 'Atoms/Command',
	parameters: { layout: 'centered' },
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj

export const Default: Story = {
	render: () => (
		<div className="w-[400px] border rounded-lg shadow-md">
			<div className="flex items-center border-b px-3">
				<Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
				<input className="flex h-11 w-full bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground" placeholder="Type a command..." />
			</div>
			<div className="p-2">
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Calendar</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Search Emoji</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Calculator</div>
			</div>
		</div>
	),
}

export const WithGroups: Story = {
	render: () => (
		<div className="w-[400px] border rounded-lg shadow-md">
			<div className="flex items-center border-b px-3">
				<Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
				<input className="flex h-11 w-full bg-transparent py-3 text-sm outline-none" placeholder="Search..." />
			</div>
			<div className="p-2">
				<div className="px-2 py-1.5 text-xs font-medium text-muted-foreground">Suggestions</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Calendar</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Calculator</div>
				<div className="px-2 py-1.5 text-xs font-medium text-muted-foreground mt-2">Settings</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Profile</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Billing</div>
			</div>
		</div>
	),
}

export const WithIcons: Story = {
	render: () => (
		<div className="w-[400px] border rounded-lg shadow-md">
			<div className="flex items-center border-b px-3">
				<Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
				<input className="flex h-11 w-full bg-transparent py-3 text-sm outline-none" placeholder="Search..." />
			</div>
			<div className="p-2">
				<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
					<Calendar className="mr-2 h-4 w-4" />
					<span>Calendar</span>
				</div>
				<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
					<User className="mr-2 h-4 w-4" />
					<span>Profile</span>
				</div>
				<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
					<Mail className="mr-2 h-4 w-4" />
					<span>Mail</span>
				</div>
				<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
					<Settings className="mr-2 h-4 w-4" />
					<span>Settings</span>
				</div>
			</div>
		</div>
	),
}

export const WithSearch: Story = {
	render: () => (
		<div className="w-[400px] border rounded-lg shadow-md">
			<div className="flex items-center border-b px-3">
				<Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
				<input className="flex h-11 w-full bg-transparent py-3 text-sm outline-none" placeholder="Type to search..." />
			</div>
			<div className="p-2">
				<div className="px-2 py-6 text-center text-sm text-muted-foreground">No results found.</div>
			</div>
		</div>
	),
}
