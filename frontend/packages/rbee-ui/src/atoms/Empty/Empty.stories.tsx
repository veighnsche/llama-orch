import type { Meta, StoryObj } from '@storybook/react'
import { Empty, EmptyHeader, EmptyTitle, EmptyDescription, EmptyContent, EmptyMedia } from './Empty'
import { Button } from '@rbee/ui/atoms/Button'
import { InboxIcon, SearchIcon, FileIcon, PlusIcon } from 'lucide-react'

const meta: Meta<typeof Empty> = {
	title: 'Atoms/Empty',
	component: Empty,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Empty>

export const Default: Story = {
	render: () => (
		<Empty>
			<EmptyHeader>
				<EmptyTitle>No results found</EmptyTitle>
				<EmptyDescription>Try adjusting your search or filter to find what you're looking for.</EmptyDescription>
			</EmptyHeader>
		</Empty>
	),
}

export const WithIcon: Story = {
	render: () => (
		<Empty>
			<EmptyHeader>
				<EmptyMedia variant="icon">
					<InboxIcon />
				</EmptyMedia>
				<EmptyTitle>No messages</EmptyTitle>
				<EmptyDescription>You don't have any messages yet. Start a conversation to see them here.</EmptyDescription>
			</EmptyHeader>
		</Empty>
	),
}

export const WithAction: Story = {
	render: () => (
		<Empty>
			<EmptyHeader>
				<EmptyMedia variant="icon">
					<FileIcon />
				</EmptyMedia>
				<EmptyTitle>No documents</EmptyTitle>
				<EmptyDescription>Get started by creating your first document.</EmptyDescription>
			</EmptyHeader>
			<EmptyContent>
				<Button>
					<PlusIcon className="mr-2 size-4" />
					Create Document
				</Button>
			</EmptyContent>
		</Empty>
	),
}

export const AllVariants: Story = {
	render: () => (
		<div className="flex flex-col gap-8 max-w-2xl">
			<Empty>
				<EmptyHeader>
					<EmptyTitle>Default Empty State</EmptyTitle>
					<EmptyDescription>Simple empty state without icon</EmptyDescription>
				</EmptyHeader>
			</Empty>

			<Empty>
				<EmptyHeader>
					<EmptyMedia variant="icon">
						<SearchIcon />
					</EmptyMedia>
					<EmptyTitle>No Search Results</EmptyTitle>
					<EmptyDescription>We couldn't find any results matching your search criteria.</EmptyDescription>
				</EmptyHeader>
				<EmptyContent>
					<Button variant="outline">Clear Search</Button>
				</EmptyContent>
			</Empty>

			<Empty>
				<EmptyHeader>
					<EmptyMedia variant="icon">
						<InboxIcon />
					</EmptyMedia>
					<EmptyTitle>Inbox Zero</EmptyTitle>
					<EmptyDescription>All caught up! You have no pending items.</EmptyDescription>
				</EmptyHeader>
			</Empty>
		</div>
	),
}
