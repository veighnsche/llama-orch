import type { Meta, StoryObj } from '@storybook/react'
import { Popover, PopoverTrigger, PopoverContent } from './Popover'
import { Button } from '@rbee/ui/atoms/Button'
import { Input } from '@rbee/ui/atoms/Input'
import { Label } from '@rbee/ui/atoms/Label'

const meta: Meta<typeof Popover> = {
	title: 'Atoms/Popover',
	component: Popover,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Popover>

export const Default: Story = {
	render: () => (
		<Popover>
			<PopoverTrigger asChild>
				<Button variant="outline">Open Popover</Button>
			</PopoverTrigger>
			<PopoverContent>
				<div className="space-y-2">
					<h4 className="font-medium leading-none">Dimensions</h4>
					<p className="text-sm text-muted-foreground">Set the dimensions for the layer.</p>
				</div>
			</PopoverContent>
		</Popover>
	),
}

export const AllPositions: Story = {
	render: () => (
		<div className="flex flex-col gap-8 items-center">
			<Popover>
				<PopoverTrigger asChild>
					<Button variant="outline">Top</Button>
				</PopoverTrigger>
				<PopoverContent side="top">
					<p className="text-sm">Content appears above</p>
				</PopoverContent>
			</Popover>

			<div className="flex gap-8">
				<Popover>
					<PopoverTrigger asChild>
						<Button variant="outline">Left</Button>
					</PopoverTrigger>
					<PopoverContent side="left">
						<p className="text-sm">Content appears to the left</p>
					</PopoverContent>
				</Popover>

				<Popover>
					<PopoverTrigger asChild>
						<Button variant="outline">Right</Button>
					</PopoverTrigger>
					<PopoverContent side="right">
						<p className="text-sm">Content appears to the right</p>
					</PopoverContent>
				</Popover>
			</div>

			<Popover>
				<PopoverTrigger asChild>
					<Button variant="outline">Bottom</Button>
				</PopoverTrigger>
				<PopoverContent side="bottom">
					<p className="text-sm">Content appears below</p>
				</PopoverContent>
			</Popover>
		</div>
	),
}

export const WithForm: Story = {
	render: () => (
		<Popover>
			<PopoverTrigger asChild>
				<Button variant="outline">Open Settings</Button>
			</PopoverTrigger>
			<PopoverContent className="w-80">
				<div className="space-y-4">
					<div className="space-y-2">
						<h4 className="font-medium leading-none">Dimensions</h4>
						<p className="text-sm text-muted-foreground">Set the dimensions for the layer.</p>
					</div>
					<div className="grid gap-2">
						<div className="grid grid-cols-3 items-center gap-4">
							<Label htmlFor="width">Width</Label>
							<Input id="width" defaultValue="100%" className="col-span-2 h-8" />
						</div>
						<div className="grid grid-cols-3 items-center gap-4">
							<Label htmlFor="maxWidth">Max. width</Label>
							<Input id="maxWidth" defaultValue="300px" className="col-span-2 h-8" />
						</div>
						<div className="grid grid-cols-3 items-center gap-4">
							<Label htmlFor="height">Height</Label>
							<Input id="height" defaultValue="25px" className="col-span-2 h-8" />
						</div>
						<div className="grid grid-cols-3 items-center gap-4">
							<Label htmlFor="maxHeight">Max. height</Label>
							<Input id="maxHeight" defaultValue="none" className="col-span-2 h-8" />
						</div>
					</div>
				</div>
			</PopoverContent>
		</Popover>
	),
}

export const WithArrow: Story = {
	render: () => (
		<Popover>
			<PopoverTrigger asChild>
				<Button variant="outline">With Arrow</Button>
			</PopoverTrigger>
			<PopoverContent>
				<div className="space-y-2">
					<h4 className="font-medium leading-none">Popover with Arrow</h4>
					<p className="text-sm text-muted-foreground">
						This popover includes an arrow pointing to the trigger element.
					</p>
				</div>
			</PopoverContent>
		</Popover>
	),
}
