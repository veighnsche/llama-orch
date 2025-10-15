// Created by: TEAM-011
import type { Meta, StoryObj } from '@storybook/react'
import { Carousel, CarouselContent, CarouselItem, CarouselNext, CarouselPrevious } from './Carousel'

const meta: Meta<typeof Carousel> = {
	title: 'Atoms/Carousel',
	component: Carousel,
	parameters: { layout: 'centered' },
	tags: ['autodocs'],
}

export default meta
type Story = StoryObj<typeof Carousel>

export const Default: Story = {
	render: () => (
		<Carousel className="w-full max-w-xs">
			<CarouselContent>
				{Array.from({ length: 5 }).map((_, index) => (
					<CarouselItem key={index}>
						<div className="p-1">
							<div className="flex aspect-square items-center justify-center p-6 border rounded-lg">
								<span className="text-4xl font-semibold">{index + 1}</span>
							</div>
						</div>
					</CarouselItem>
				))}
			</CarouselContent>
			<CarouselPrevious />
			<CarouselNext />
		</Carousel>
	),
}

export const WithThumbnails: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<Carousel className="w-full max-w-xs">
				<CarouselContent>
					{Array.from({ length: 5 }).map((_, index) => (
						<CarouselItem key={index}>
							<div className="flex aspect-square items-center justify-center p-6 border rounded-lg bg-muted">
								<span className="text-2xl">Slide {index + 1}</span>
							</div>
						</CarouselItem>
					))}
				</CarouselContent>
				<CarouselPrevious />
				<CarouselNext />
			</Carousel>
		</div>
	),
}

export const Autoplay: Story = {
	render: () => (
		<Carousel className="w-full max-w-xs" opts={{ loop: true }}>
			<CarouselContent>
				{Array.from({ length: 5 }).map((_, index) => (
					<CarouselItem key={index}>
						<div className="flex aspect-square items-center justify-center p-6 border rounded-lg">
							<span className="text-4xl font-semibold">{index + 1}</span>
						</div>
					</CarouselItem>
				))}
			</CarouselContent>
			<CarouselPrevious />
			<CarouselNext />
		</Carousel>
	),
}

export const Vertical: Story = {
	render: () => (
		<Carousel orientation="vertical" className="w-full max-w-xs">
			<CarouselContent className="h-[200px]">
				{Array.from({ length: 5 }).map((_, index) => (
					<CarouselItem key={index}>
						<div className="p-1">
							<div className="flex items-center justify-center p-6 border rounded-lg">
								<span className="text-3xl font-semibold">{index + 1}</span>
							</div>
						</div>
					</CarouselItem>
				))}
			</CarouselContent>
			<CarouselPrevious />
			<CarouselNext />
		</Carousel>
	),
}
