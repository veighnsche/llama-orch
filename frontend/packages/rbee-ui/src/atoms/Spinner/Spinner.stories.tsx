// Created by: TEAM-008
import type { Meta, StoryObj } from '@storybook/react'
import { Spinner } from './Spinner'

const meta: Meta<typeof Spinner> = {
	title: 'Atoms/Spinner',
	component: Spinner,
	parameters: {
		layout: 'centered',
		docs: {
			description: {
				component: `
A loading spinner component using Lucide's Loader2 icon with animation.

## Features
- Smooth rotation animation
- Accessible with role="status" and aria-label
- Respects currentColor for theming
- Default size of 16px (size-4)
- Customizable size and color

## Used In
- Loading states
- Buttons during async operations
- Page loaders
- Skeleton screens
				`,
			},
		},
	},
	tags: ['autodocs'],
	argTypes: {
		className: {
			control: 'text',
			description: 'Additional CSS classes for size and color',
		},
	},
}

export default meta
type Story = StoryObj<typeof Spinner>

export const Default: Story = {
	args: {},
}

export const AllSizes: Story = {
	render: () => (
		<div className="flex items-end gap-8">
			<div className="flex flex-col items-center gap-2">
				<Spinner className="size-3" />
				<span className="text-xs text-muted-foreground">12px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<Spinner />
				<span className="text-xs text-muted-foreground">16px (default)</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<Spinner className="size-5" />
				<span className="text-xs text-muted-foreground">20px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<Spinner className="size-6" />
				<span className="text-xs text-muted-foreground">24px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<Spinner className="size-8" />
				<span className="text-xs text-muted-foreground">32px</span>
			</div>
			<div className="flex flex-col items-center gap-2">
				<Spinner className="size-12" />
				<span className="text-xs text-muted-foreground">48px</span>
			</div>
		</div>
	),
}

export const AllColors: Story = {
	render: () => (
		<div className="flex flex-col gap-6">
			<div className="flex items-center gap-4">
				<Spinner className="text-foreground" />
				<span className="text-sm">Foreground (default)</span>
			</div>
			<div className="flex items-center gap-4">
				<Spinner className="text-muted-foreground" />
				<span className="text-sm">Muted</span>
			</div>
			<div className="flex items-center gap-4">
				<Spinner className="text-primary" />
				<span className="text-sm">Primary</span>
			</div>
			<div className="flex items-center gap-4">
				<Spinner className="text-destructive" />
				<span className="text-sm">Destructive</span>
			</div>
			<div className="flex items-center gap-4">
				<Spinner className="text-blue-500" />
				<span className="text-sm">Custom (Blue)</span>
			</div>
			<div className="flex items-center gap-4">
				<Spinner className="text-green-500" />
				<span className="text-sm">Custom (Green)</span>
			</div>
		</div>
	),
}

export const InButton: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<div className="flex gap-3">
				<button className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium">
					<Spinner className="size-4" />
					Loading...
				</button>
				<button
					disabled
					className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium opacity-50 cursor-not-allowed"
				>
					<Spinner className="size-4" />
					Processing
				</button>
			</div>

			<div className="flex gap-3">
				<button className="inline-flex items-center gap-2 px-4 py-2 border border-input rounded-md text-sm font-medium hover:bg-muted/40 transition-colors">
					<Spinner className="size-4" />
					Loading...
				</button>
				<button className="inline-flex items-center gap-2 px-3 py-1.5 border border-input rounded-md text-xs font-medium hover:bg-muted/40 transition-colors">
					<Spinner className="size-3" />
					Loading
				</button>
			</div>

			<div className="flex gap-3">
				<button className="inline-flex items-center justify-center size-9 rounded-lg border border-input hover:bg-muted/40 transition-colors">
					<Spinner className="size-4" />
				</button>
				<button className="inline-flex items-center justify-center size-10 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors">
					<Spinner className="size-5" />
				</button>
			</div>
		</div>
	),
}

export const LoadingStates: Story = {
	render: () => (
		<div className="flex flex-col gap-6 w-full max-w-md">
			<div className="p-6 border border-border rounded-lg">
				<div className="flex items-center gap-3 mb-4">
					<Spinner className="size-5" />
					<h3 className="text-sm font-semibold">Loading data...</h3>
				</div>
				<p className="text-sm text-muted-foreground">Please wait while we fetch your information.</p>
			</div>

			<div className="p-6 border border-border rounded-lg">
				<div className="flex flex-col items-center gap-4 py-8">
					<Spinner className="size-8" />
					<div className="text-center">
						<h3 className="text-sm font-semibold mb-1">Deploying model</h3>
						<p className="text-sm text-muted-foreground">This may take a few minutes...</p>
					</div>
				</div>
			</div>

			<div className="p-4 border border-border rounded-lg">
				<div className="flex items-center justify-between">
					<div className="flex items-center gap-3">
						<Spinner className="size-4 text-primary" />
						<span className="text-sm">Uploading files</span>
					</div>
					<span className="text-sm text-muted-foreground">45%</span>
				</div>
			</div>
		</div>
	),
}

export const InlineLoading: Story = {
	render: () => (
		<div className="flex flex-col gap-4 max-w-md">
			<p className="text-sm">
				<Spinner className="inline size-4 mr-2" />
				Loading your dashboard...
			</p>

			<p className="text-sm">
				Processing your request
				<Spinner className="inline size-3 ml-2" />
			</p>

			<div className="flex items-center gap-2 text-sm text-muted-foreground">
				<Spinner className="size-3" />
				<span>Syncing data in the background</span>
			</div>
		</div>
	),
}

export const FullPageLoader: Story = {
	render: () => (
		<div className="w-full max-w-4xl h-[400px] border border-border rounded-lg relative overflow-hidden">
			<div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm">
				<div className="flex flex-col items-center gap-4">
					<Spinner className="size-12 text-primary" />
					<div className="text-center">
						<h3 className="text-lg font-semibold mb-1">Loading Application</h3>
						<p className="text-sm text-muted-foreground">Initializing your workspace...</p>
					</div>
				</div>
			</div>
		</div>
	),
}

export const ListLoading: Story = {
	render: () => (
		<div className="w-full max-w-md space-y-3">
			<div className="flex items-center gap-3 p-3 border border-border rounded-lg">
				<Spinner className="size-4" />
				<div className="flex-1">
					<div className="h-4 bg-muted rounded w-3/4 mb-2" />
					<div className="h-3 bg-muted rounded w-1/2" />
				</div>
			</div>
			<div className="flex items-center gap-3 p-3 border border-border rounded-lg">
				<Spinner className="size-4" />
				<div className="flex-1">
					<div className="h-4 bg-muted rounded w-2/3 mb-2" />
					<div className="h-3 bg-muted rounded w-1/3" />
				</div>
			</div>
			<div className="flex items-center gap-3 p-3 border border-border rounded-lg">
				<Spinner className="size-4" />
				<div className="flex-1">
					<div className="h-4 bg-muted rounded w-4/5 mb-2" />
					<div className="h-3 bg-muted rounded w-2/5" />
				</div>
			</div>
		</div>
	),
}
