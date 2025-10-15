import type { Meta, StoryObj } from '@storybook/react'
import {
	Item,
	ItemMedia,
	ItemContent,
	ItemActions,
	ItemGroup,
	ItemSeparator,
	ItemTitle,
	ItemDescription,
} from './Item'
import { Button } from '@rbee/ui/atoms/Button'
import { Badge } from '@rbee/ui/atoms/Badge'
import { FileIcon, FolderIcon, ImageIcon, MoreHorizontalIcon } from 'lucide-react'

const meta: Meta<typeof Item> = {
	title: 'Atoms/Item',
	component: Item,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		variant: {
			control: 'select',
			options: ['default', 'outline', 'muted'],
		},
		size: {
			control: 'select',
			options: ['default', 'sm'],
		},
	},
}

export default meta
type Story = StoryObj<typeof Item>

export const Default: Story = {
	render: () => (
		<Item className="w-96">
			<ItemContent>
				<ItemTitle>Project Files</ItemTitle>
				<ItemDescription>All your project files and documents</ItemDescription>
			</ItemContent>
		</Item>
	),
}

export const WithIcon: Story = {
	render: () => (
		<Item className="w-96">
			<ItemMedia variant="icon">
				<FileIcon />
			</ItemMedia>
			<ItemContent>
				<ItemTitle>Document.pdf</ItemTitle>
				<ItemDescription>Last modified 2 hours ago</ItemDescription>
			</ItemContent>
		</Item>
	),
}

export const WithDescription: Story = {
	render: () => (
		<Item className="w-96">
			<ItemMedia variant="icon">
				<FolderIcon />
			</ItemMedia>
			<ItemContent>
				<ItemTitle>Project Folder</ItemTitle>
				<ItemDescription>
					Contains all project files, documentation, and assets. Last updated today at 3:45 PM.
				</ItemDescription>
			</ItemContent>
			<ItemActions>
				<Button variant="ghost" size="icon-sm">
					<MoreHorizontalIcon />
				</Button>
			</ItemActions>
		</Item>
	),
}

export const Selected: Story = {
	render: () => (
		<ItemGroup className="w-96">
			<Item variant="muted">
				<ItemMedia variant="icon">
					<FileIcon />
				</ItemMedia>
				<ItemContent>
					<ItemTitle>Selected Document.pdf</ItemTitle>
					<ItemDescription>Currently selected item</ItemDescription>
				</ItemContent>
				<ItemActions>
					<Badge>Selected</Badge>
				</ItemActions>
			</Item>
			<ItemSeparator />
			<Item>
				<ItemMedia variant="icon">
					<ImageIcon />
				</ItemMedia>
				<ItemContent>
					<ItemTitle>Image.png</ItemTitle>
					<ItemDescription>Not selected</ItemDescription>
				</ItemContent>
			</Item>
			<ItemSeparator />
			<Item>
				<ItemMedia variant="icon">
					<FolderIcon />
				</ItemMedia>
				<ItemContent>
					<ItemTitle>Folder</ItemTitle>
					<ItemDescription>Not selected</ItemDescription>
				</ItemContent>
			</Item>
		</ItemGroup>
	),
}
