// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Copy, Cut, Paste, Download, Share } from 'lucide-react'

const meta: Meta = {
	title: 'Atoms/ContextMenu',
	parameters: { layout: 'centered' },
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj

export const Default: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<div className="border rounded-lg p-8 text-center bg-muted">
				<p className="text-sm text-muted-foreground">Right-click here to open context menu</p>
			</div>
			<div className="w-[200px] border rounded-lg shadow-lg bg-popover p-1">
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Copy</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Cut</div>
				<div className="px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">Paste</div>
			</div>
		</div>
	),
}

export const WithIcons: Story = {
	render: () => (
		<div className="w-[200px] border rounded-lg shadow-lg bg-popover p-1">
			<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<Copy className="mr-2 h-4 w-4" />
				<span>Copy</span>
			</div>
			<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<Cut className="mr-2 h-4 w-4" />
				<span>Cut</span>
			</div>
			<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<Paste className="mr-2 h-4 w-4" />
				<span>Paste</span>
			</div>
		</div>
	),
}

export const WithSubmenus: Story = {
	render: () => (
		<div className="w-[200px] border rounded-lg shadow-lg bg-popover p-1">
			<div className="flex items-center justify-between px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<span>Share</span>
				<Share className="h-4 w-4" />
			</div>
			<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<Download className="mr-2 h-4 w-4" />
				<span>Download</span>
			</div>
		</div>
	),
}

export const WithSeparators: Story = {
	render: () => (
		<div className="w-[200px] border rounded-lg shadow-lg bg-popover p-1">
			<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<Copy className="mr-2 h-4 w-4" />
				<span>Copy</span>
			</div>
			<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<Cut className="mr-2 h-4 w-4" />
				<span>Cut</span>
			</div>
			<div className="h-px bg-border my-1" />
			<div className="flex items-center px-2 py-1.5 text-sm hover:bg-accent rounded-sm cursor-pointer">
				<Paste className="mr-2 h-4 w-4" />
				<span>Paste</span>
			</div>
			<div className="h-px bg-border my-1" />
			<div className="flex items-center px-2 py-1.5 text-sm text-destructive hover:bg-accent rounded-sm cursor-pointer">
				<span>Delete</span>
			</div>
		</div>
	),
}
