import { cn } from '@rbee/ui/utils'
import { Star } from 'lucide-react'

export interface RatingStarsProps {
	rating: number
	size?: 'sm' | 'md'
	className?: string
}

export function RatingStars({ rating, size = 'md', className }: RatingStarsProps) {
	const sizeClasses = {
		sm: 'h-3.5 w-3.5',
		md: 'h-4 w-4',
	}

	return (
		<div className={cn('flex items-center gap-0.5', className)} aria-label={`${rating} out of 5 stars`}>
			{Array.from({ length: 5 }).map((_, i) => (
				<Star
					key={i}
					className={cn(sizeClasses[size], i < rating ? 'fill-primary text-primary' : 'fill-muted text-muted')}
					aria-hidden="true"
				/>
			))}
			<span className="sr-only">{rating} out of 5 stars</span>
		</div>
	)
}
