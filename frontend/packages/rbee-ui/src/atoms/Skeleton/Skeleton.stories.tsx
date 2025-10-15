import type { Meta, StoryObj } from '@storybook/react'
import { Skeleton } from './Skeleton'

const meta: Meta<typeof Skeleton> = {
	title: 'Atoms/Skeleton',
	component: Skeleton,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Skeleton>

export const Default: Story = {
	render: () => <Skeleton className="h-12 w-[250px]" />,
}

export const AllShapes: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<Skeleton className="h-4 w-[250px]" />
			<Skeleton className="h-4 w-[200px]" />
			<Skeleton className="h-4 w-[150px]" />
			<Skeleton className="size-12 rounded-full" />
			<Skeleton className="h-32 w-[300px] rounded-lg" />
		</div>
	),
}

export const Card: Story = {
	render: () => (
		<div className="flex w-[350px] flex-col gap-3 rounded-lg border p-4">
			<div className="flex items-center gap-4">
				<Skeleton className="size-12 rounded-full" />
				<div className="flex-1 space-y-2">
					<Skeleton className="h-4 w-[200px]" />
					<Skeleton className="h-4 w-[150px]" />
				</div>
			</div>
			<Skeleton className="h-[200px] w-full" />
			<div className="space-y-2">
				<Skeleton className="h-4 w-full" />
				<Skeleton className="h-4 w-full" />
				<Skeleton className="h-4 w-3/4" />
			</div>
		</div>
	),
}

export const List: Story = {
	render: () => (
		<div className="w-[400px] space-y-4">
			{Array.from({ length: 5 }).map((_, i) => (
				<div key={i} className="flex items-center gap-4">
					<Skeleton className="size-10 rounded-full" />
					<div className="flex-1 space-y-2">
						<Skeleton className="h-4 w-full" />
						<Skeleton className="h-3 w-3/4" />
					</div>
				</div>
			))}
		</div>
	),
}
