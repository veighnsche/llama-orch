import type { Meta, StoryObj } from '@storybook/react'
import { RatingStars } from './RatingStars'
import { useState } from 'react'
import { Star } from 'lucide-react'
import { cn } from '@rbee/ui/utils'

const meta: Meta<typeof RatingStars> = {
	title: 'Atoms/RatingStars',
	component: RatingStars,
	parameters: {
		layout: 'centered',
	},
	tags: ['autodocs'],
	argTypes: {
		rating: {
			control: { type: 'number', min: 0, max: 5, step: 0.5 },
		},
		size: {
			control: 'select',
			options: ['sm', 'md'],
		},
	},
}

export default meta
type Story = StoryObj<typeof RatingStars>

export const Default: Story = {
	args: {
		rating: 4,
		size: 'md',
	},
}

export const AllRatings: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<div className="flex items-center gap-2">
				<RatingStars rating={5} />
				<span className="text-sm text-muted-foreground">5 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={4} />
				<span className="text-sm text-muted-foreground">4 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={3} />
				<span className="text-sm text-muted-foreground">3 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={2} />
				<span className="text-sm text-muted-foreground">2 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={1} />
				<span className="text-sm text-muted-foreground">1 star</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={0} />
				<span className="text-sm text-muted-foreground">0 stars</span>
			</div>
		</div>
	),
}

export const HalfStars: Story = {
	render: () => (
		<div className="flex flex-col gap-4">
			<div className="flex items-center gap-2">
				<RatingStars rating={4.5} />
				<span className="text-sm text-muted-foreground">4.5 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={3.5} />
				<span className="text-sm text-muted-foreground">3.5 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={2.5} />
				<span className="text-sm text-muted-foreground">2.5 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={1.5} />
				<span className="text-sm text-muted-foreground">1.5 stars</span>
			</div>
			<div className="flex items-center gap-2">
				<RatingStars rating={0.5} />
				<span className="text-sm text-muted-foreground">0.5 stars</span>
			</div>
		</div>
	),
}

export const Interactive: Story = {
	render: () => {
		const [rating, setRating] = useState(0)
		const [hover, setHover] = useState(0)

		return (
			<div className="flex flex-col gap-4 items-center">
				<div className="flex items-center gap-0.5">
					{Array.from({ length: 5 }).map((_, i) => (
						<button
							key={i}
							type="button"
							onClick={() => setRating(i + 1)}
							onMouseEnter={() => setHover(i + 1)}
							onMouseLeave={() => setHover(0)}
							className="cursor-pointer transition-transform hover:scale-110"
						>
							<Star
								className={cn(
									'h-6 w-6',
									i < (hover || rating) ? 'fill-primary text-primary' : 'fill-muted text-muted',
								)}
							/>
						</button>
					))}
				</div>
				<p className="text-sm text-muted-foreground">
					{rating > 0 ? `You rated ${rating} out of 5 stars` : 'Click to rate'}
				</p>
			</div>
		)
	},
}
