import type { Meta, StoryObj } from '@storybook/react'
import { HoverCard, HoverCardTrigger, HoverCardContent } from './HoverCard'
import { Button } from '@rbee/ui/atoms/Button'
import { Avatar, AvatarImage, AvatarFallback } from '@rbee/ui/atoms/Avatar'
import { CalendarIcon } from 'lucide-react'

const meta: Meta<typeof HoverCard> = {
	title: 'Atoms/HoverCard',
	component: HoverCard,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof HoverCard>

export const Default: Story = {
	render: () => (
		<HoverCard>
			<HoverCardTrigger asChild>
				<Button variant="link">@nextjs</Button>
			</HoverCardTrigger>
			<HoverCardContent>
				<div className="space-y-2">
					<h4 className="text-sm font-semibold">@nextjs</h4>
					<p className="text-sm text-muted-foreground">
						The React Framework – created and maintained by @vercel.
					</p>
					<div className="flex items-center pt-2">
						<CalendarIcon className="mr-2 size-4 opacity-70" />
						<span className="text-xs text-muted-foreground">Joined December 2021</span>
					</div>
				</div>
			</HoverCardContent>
		</HoverCard>
	),
}

export const WithImage: Story = {
	render: () => (
		<HoverCard>
			<HoverCardTrigger asChild>
				<Button variant="link">@shadcn</Button>
			</HoverCardTrigger>
			<HoverCardContent className="w-80">
				<div className="flex justify-between space-x-4">
					<Avatar>
						<AvatarImage src="https://github.com/shadcn.png" />
						<AvatarFallback>SC</AvatarFallback>
					</Avatar>
					<div className="space-y-1">
						<h4 className="text-sm font-semibold">@shadcn</h4>
						<p className="text-sm text-muted-foreground">
							The React Framework for the Web – created and maintained by @vercel.
						</p>
						<div className="flex items-center pt-2">
							<CalendarIcon className="mr-2 size-4 opacity-70" />
							<span className="text-xs text-muted-foreground">Joined December 2021</span>
						</div>
					</div>
				</div>
			</HoverCardContent>
		</HoverCard>
	),
}

export const AllPositions: Story = {
	render: () => (
		<div className="flex flex-col gap-8 items-center">
			<HoverCard>
				<HoverCardTrigger asChild>
					<Button variant="outline">Hover (Top)</Button>
				</HoverCardTrigger>
				<HoverCardContent side="top">
					<p className="text-sm">Content appears above the trigger</p>
				</HoverCardContent>
			</HoverCard>

			<div className="flex gap-8">
				<HoverCard>
					<HoverCardTrigger asChild>
						<Button variant="outline">Hover (Left)</Button>
					</HoverCardTrigger>
					<HoverCardContent side="left">
						<p className="text-sm">Content appears to the left</p>
					</HoverCardContent>
				</HoverCard>

				<HoverCard>
					<HoverCardTrigger asChild>
						<Button variant="outline">Hover (Right)</Button>
					</HoverCardTrigger>
					<HoverCardContent side="right">
						<p className="text-sm">Content appears to the right</p>
					</HoverCardContent>
				</HoverCard>
			</div>

			<HoverCard>
				<HoverCardTrigger asChild>
					<Button variant="outline">Hover (Bottom)</Button>
				</HoverCardTrigger>
				<HoverCardContent side="bottom">
					<p className="text-sm">Content appears below the trigger</p>
				</HoverCardContent>
			</HoverCard>
		</div>
	),
}

export const WithDelay: Story = {
	render: () => (
		<div className="flex gap-4">
			<HoverCard openDelay={0}>
				<HoverCardTrigger asChild>
					<Button variant="outline">No Delay</Button>
				</HoverCardTrigger>
				<HoverCardContent>
					<p className="text-sm">Opens immediately</p>
				</HoverCardContent>
			</HoverCard>

			<HoverCard openDelay={500}>
				<HoverCardTrigger asChild>
					<Button variant="outline">500ms Delay</Button>
				</HoverCardTrigger>
				<HoverCardContent>
					<p className="text-sm">Opens after 500ms</p>
				</HoverCardContent>
			</HoverCard>

			<HoverCard openDelay={1000}>
				<HoverCardTrigger asChild>
					<Button variant="outline">1000ms Delay</Button>
				</HoverCardTrigger>
				<HoverCardContent>
					<p className="text-sm">Opens after 1 second</p>
				</HoverCardContent>
			</HoverCard>
		</div>
	),
}
