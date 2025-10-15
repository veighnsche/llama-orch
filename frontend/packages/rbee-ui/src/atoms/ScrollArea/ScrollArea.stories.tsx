import type { Meta, StoryObj } from '@storybook/react'
import { ScrollArea } from './ScrollArea'

const meta: Meta<typeof ScrollArea> = {
	title: 'Atoms/ScrollArea',
	component: ScrollArea,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof ScrollArea>

export const Default: Story = {
	render: () => (
		<ScrollArea className="h-[200px] w-[350px] rounded-md border p-4">
			<div className="space-y-4">
				{Array.from({ length: 20 }).map((_, i) => (
					<div key={i} className="text-sm">
						Item {i + 1}: This is a scrollable content area with vertical scrolling enabled.
					</div>
				))}
			</div>
		</ScrollArea>
	),
}

export const Horizontal: Story = {
	render: () => (
		<ScrollArea className="h-[100px] w-[350px] rounded-md border p-4">
			<div className="flex gap-4" style={{ width: '800px' }}>
				{Array.from({ length: 10 }).map((_, i) => (
					<div key={i} className="flex-shrink-0 rounded-md bg-muted p-4 text-sm">
						Column {i + 1}
					</div>
				))}
			</div>
		</ScrollArea>
	),
}

export const Both: Story = {
	render: () => (
		<ScrollArea className="h-[200px] w-[350px] rounded-md border p-4">
			<div className="space-y-4" style={{ width: '600px' }}>
				{Array.from({ length: 15 }).map((_, i) => (
					<div key={i} className="text-sm">
						Row {i + 1}: This content is wide enough to require horizontal scrolling as well as vertical
						scrolling to see all content.
					</div>
				))}
			</div>
		</ScrollArea>
	),
}

export const WithShadows: Story = {
	render: () => (
		<div className="relative">
			<ScrollArea className="h-[300px] w-[400px] rounded-md border p-4">
				<div className="space-y-4">
					<h3 className="text-lg font-semibold">Scrollable Content</h3>
					{Array.from({ length: 30 }).map((_, i) => (
						<p key={i} className="text-sm">
							Paragraph {i + 1}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
							tempor incididunt ut labore et dolore magna aliqua.
						</p>
					))}
				</div>
			</ScrollArea>
		</div>
	),
}
